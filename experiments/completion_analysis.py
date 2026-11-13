"""Offline, provenance-explicit six-method aggregation; never invokes fitting."""
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .completion_study import ROOT, OUT, METHODS, acceptable, inventory, candidates
from .simulation_analysis import point_metrics, interval_metrics, paired_comparison, _csv
from .simulation_config import digest
from .simulation_storage import commit_bytes, commit_json, read_json, publish_pointer

LABELS = dict(zip(METHODS, ['MLE U', 'MLE R', 'Independent U', 'Independent R', 'Hierarchical U', 'Hierarchical R']))
PRIMARY = ['a', 'b1[A]', 'b2[A]', 'b1[B]', 'b2[B]']


def common_ids(points, methods=METHODS):
    sets = [{p['dataset_id'] for p in points if p['method'] == m and p['valid_point']}
            for m in methods]
    return sorted(set.intersection(*sets))


def analyze(cfg):
    inv = inventory(cfg)
    points, intervals, diags, times, selected, attempts, issues = [], [], [], [], [], [], []
    dataset_records = {}
    scenarios = {s['id']: s for s in cfg['scenarios']}
    for row in inv:
        s = scenarios[row['scenario_id']]
        meta, sha, found = candidates(cfg, s, row['repetition'], row['method'])
        dataset_records[meta['dataset_id']] = dict(dataset_id=meta['dataset_id'],
            scenario_id=meta['scenario_id'], repetition=meta['repetition'], dataset_sha256=sha,
            **meta['seed_information'])
        for path, b in found:
            attempts.append(dict(dataset_id=meta['dataset_id'], method=b['method'],
                path=str(path), status=b['status'], acceptable=acceptable(b),
                runtime_seconds=b['runtime_seconds'], mcmc=b['mcmc']))
        if row['selected_path'] is None:
            issues.append(dict(**row, problem='not_attempted'))
            continue
        b, file_sha = read_json(row['selected_path'])
        good = acceptable(b)
        if not good:
            issues.append(dict(**row, problem=b['status']))
        context = {k: b[k] for k in ('dataset_id', 'dataset_sha256', 'scenario_id', 'repetition', 'method', 'fit_id')}
        context.update(selected_path=row['selected_path'], selected_sha256=file_sha)
        selected.append(dict(**context, valid=good, status=b['status']))
        times.append(dict(**context, valid=good, runtime_seconds=b['runtime_seconds'], mcmc=b['mcmc']))
        for p in b['points']:
            assert p['truth'] >= 0
            points.append(dict(**context, **p))
        truths = {p['parameter']: p['truth'] for p in b['points']}
        for i in b['intervals']:
            if i['valid_interval']:
                assert i['status'] == 'empty' or (np.isfinite(i['lower']) and np.isfinite(i['upper']) and i['lower'] <= i['upper'])
                if i['status'] != 'empty':
                    assert i['covered'] == int(i['lower'] <= truths[i['parameter']] <= i['upper'])
                    assert np.isclose(i['width'], i['upper']-i['lower'])
                    if i['lower'] < 0:
                        issues.append(dict(**context, problem='negative_raw_wald_endpoint', parameter=i['parameter'], family=i['interval_family']))
            intervals.append(dict(**context, **i, truth=truths[i['parameter']]))
        for d in b['bayesian_diagnostics']:
            diags.append(dict(**context, **d, valid=good))
        artifact = b.get('draw_artifact')
        if artifact:
            path = ROOT/artifact['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
            with np.load(path, allow_pickle=False) as f:
                draws, names = f['draws'], list(f['names'])
            assert np.all(np.isfinite(draws)) and np.all(draws > 0)
            assert list(draws.shape) == artifact['shape']
            if b['restricted']:
                for block in ('A', 'B'):
                    assert np.all(draws[:, :, names.index(f'b1[{block}]')] <= draws[:, :, names.index(f'b2[{block}]')])
            if good:
                for p in b['points']:
                    if p['parameter'] in names:
                        assert np.isclose(p['estimate'], draws[:, :, names.index(p['parameter'])].mean(), rtol=1e-12)
    assert len({(r['dataset_id'], r['method']) for r in selected}) == len(selected)
    snapshot = digest(dict(selected=selected, attempts=attempts,
        sources={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                 ('experiments/completion_analysis.py', 'experiments/completion_publication.py',
                  'experiments/completion_study.py', 'experiments/completion_qc.py')}))[:24]
    dest = OUT/'analysis'/snapshot
    dest.mkdir(parents=True, exist_ok=True)
    tables = dict(datasets=list(dataset_records.values()), inventory=inv, selected_fits=selected, attempt_history=attempts,
                  replication_estimates=points, replication_intervals=intervals,
                  bayesian_diagnostics=diags, runtime=times, issues=issues)
    performance, iperf, paired, pairwise, ranks, memberships = [], [], [], [], [], []
    for sid in scenarios:
        sp = [r for r in points if r['scenario_id'] == sid]
        for name in sorted({r['parameter'] for r in sp}):
            rows = [r for r in sp if r['parameter'] == name]
            ids = common_ids(rows)
            memberships.append(dict(scenario_id=sid, parameter=name, R_paired=len(ids), dataset_ids=ids))
            for method in METHODS:
                mr = [r for r in rows if r['method'] == method]
                assert len({r['dataset_id'] for r in mr}) == len(mr)
                context = dict(scenario_id=sid, method=method, parameter=name, truth=mr[0]['truth'] if mr else None)
                performance.append(dict(**context, **point_metrics(mr)))
                metric = point_metrics([r for r in mr if r['dataset_id'] in ids])
                paired.append(dict(**context, **metric, dataset_ids=ids))
            subset = [r for r in paired if r['scenario_id'] == sid and r['parameter'] == name and r['mse'] is not None]
            ordered = sorted(subset, key=lambda r: r['mse'])
            for r in subset:
                ranks.append(dict(scenario_id=sid, parameter=name, method=r['method'], mse=r['mse'],
                    rank=1+sum(o['mse'] < r['mse'] for o in ordered), R_paired=len(ids),
                    limitation='Descriptive per-parameter ranking; only 20 independent datasets per scenario.'))
            # All 15 contrasts use the full-six common IDs, never pair-specific subsets.
            for u, v in combinations(METHODS, 2):
                ur = [r for r in rows if r['method'] == u and r['dataset_id'] in ids]
                vr = [r for r in rows if r['method'] == v and r['dataset_id'] in ids]
                pairwise.append(dict(scenario_id=sid, parameter=name, method_U=u, method_R=v,
                    dataset_ids=ids, **paired_comparison(ur, vr)))
    frame = pd.DataFrame(intervals)
    for keys, group in frame.groupby(['scenario_id', 'method', 'parameter', 'interval_family', 'interval_kind']):
        context = dict(zip(['scenario_id', 'method', 'parameter', 'interval_family', 'interval_kind'], keys))
        iperf.append(dict(**context, **interval_metrics(group.to_dict('records'))))
    counts = []
    for sid in scenarios:
        for m in METHODS:
            rr = [r for r in inv if r['scenario_id'] == sid and r['method'] == m]
            counts.append(dict(scenario_id=sid, method=m, planned=20,
                valid=sum(r['state'] == 'complete' for r in rr),
                invalid=sum(r['state'] == 'invalid' for r in rr), missing=sum(r['state'] == 'missing' for r in rr)))
    runtime_summary = []
    for m in METHODS:
        tt = [r['runtime_seconds'] for r in times if r['method'] == m]
        aa = [r['runtime_seconds'] for r in attempts if r['method'] == m]
        runtime_summary.append(dict(method=m, selected_fits=len(tt), mean_selected_seconds=np.mean(tt),
            median_selected_seconds=np.median(tt), total_selected_seconds=sum(tt),
            total_all_attempt_seconds=sum(aa),
            note='Includes historical and new implementations/chain lengths; excludes I/O and supervisor overhead.'))
    tables.update(parameter_performance=performance, interval_performance=iperf,
        six_method_paired=paired, paired_dataset_ids=memberships, relative_mse_comparisons=pairwise,
        rankings=ranks, valid_fit_counts=counts, runtime_summary=runtime_summary)
    ds = pd.DataFrame(diags)
    tables['mcmc_summary'] = [dict(method=m, parameter=p, fits=len(g),
        maximum_rhat=g.rhat.max(), minimum_bulk_ess=g.ess_bulk.min(),
        minimum_tail_ess=g.ess_tail.min(), maximum_mean_mcse=g.mcse_mean.max())
        for (m, p), g in ds.groupby(['method', 'parameter'])]
    joined = ds.merge(pd.DataFrame(points)[['dataset_id', 'method', 'parameter', 'estimate', 'target_kind']],
                      on=['dataset_id', 'method', 'parameter'])
    tables['mcse_summary'] = [dict(method=m, target_kind=kind, monitored_entries=len(g),
        maximum_relative_mean_mcse=float((g.mcse_mean/g.estimate.abs()).max()),
        note='Relative MCSE is MCSE/abs(posterior mean); small tail probabilities can inflate this ratio.')
        for (m, kind), g in joined.groupby(['method', 'target_kind'])]
    for name, rows in tables.items():
        commit_bytes(dest/(name+'.csv'), _csv(rows))
    markdown_columns = {
        'valid_fit_counts': ['scenario_id', 'method', 'planned', 'valid', 'invalid', 'missing'],
        'parameter_performance': ['scenario_id', 'method', 'parameter', 'R_point', 'absolute_bias', 'mse'],
        'six_method_paired': ['scenario_id', 'method', 'parameter', 'R_point', 'absolute_bias', 'mse'],
        'interval_performance': ['scenario_id', 'method', 'parameter', 'interval_family', 'interval_kind', 'R_CP', 'R_AW', 'coverage_probability', 'average_width'],
        'relative_mse_comparisons': ['scenario_id', 'parameter', 'method_U', 'method_R', 'R_paired', 'MSE_U_paired', 'MSE_R_paired', 'relative_mse_reduction_percent', 'paired_difference_mcse'],
        'rankings': ['scenario_id', 'parameter', 'method', 'rank', 'mse', 'R_paired'],
        'runtime_summary': ['method', 'selected_fits', 'mean_selected_seconds', 'median_selected_seconds', 'total_all_attempt_seconds'],
        'mcmc_summary': ['method', 'parameter', 'fits', 'maximum_rhat', 'minimum_bulk_ess', 'minimum_tail_ess', 'maximum_mean_mcse'],
        'mcse_summary': ['method', 'target_kind', 'maximum_relative_mean_mcse'],
    }
    (dest/'markdown').mkdir(exist_ok=True)
    for name, columns in markdown_columns.items():
        def cell(value):
            if value is None: return 'NA'
            if isinstance(value, float): return f'{value:.12g}'
            return str(value).replace('|', '\\|')
        lines = ['# '+name.replace('_', ' '), '',
            'CSV is authoritative at full round-trip precision. R columns are the explicit metric denominators. '
            'MSE ranks are descriptive and scenario/parameter specific; only 20 independent replications per scenario.', '',
            '| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        lines += ['| '+' | '.join(cell(r.get(c)) for c in columns)+' |' for r in tables[name]]
        (dest/'markdown'/(name+'.md')).write_text('\n'.join(lines)+'\n')
    validation = dict(planned=360, selected=len(selected), valid=sum(r['valid'] for r in selected),
        original_datasets=60, unique_dataset_ids=len({r['dataset_id'] for r in inv}),
        missing=sum(r['state'] == 'missing' for r in inv), invalid=sum(r['state'] == 'invalid' for r in inv),
        counts=counts, snapshot=snapshot, all_six_memberships=memberships,
        checks=['dataset envelope/identity/seed/design hashes', 'no duplicate selected fits',
                'all primary estimates/support/moments', 'parameter-wise Rhat and bulk/tail ESS',
                'interval endpoints/coverage/width', 'draw hashes/support/posterior means',
                'six-way actual dataset-ID intersection', 'all attempts preserved'])
    commit_json(dest/'validation.json', validation)
    plots(dest, tables)
    write_summary(dest, tables, validation)
    from .completion_publication import write_manuscript
    write_manuscript(dest, tables, validation)
    publish_pointer(OUT/'latest', dest)
    from .completion_qc import verify
    verify(dest)
    print(json.dumps({k: validation[k] for k in ('snapshot', 'valid', 'missing', 'invalid')}, indent=2))
    print(dest)


def plots(dest, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    figures = dest/'figures'
    figures.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(figures/(name+'.png'), bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    perf = pd.DataFrame(tables['six_method_paired'])
    scenarios = list(perf.scenario_id.unique())
    for metric, title in [('mse', 'Mean squared error'), ('absolute_bias', 'Absolute bias')]:
        fig, axes = plt.subplots(3, 5, figsize=(17, 10), squeeze=False, constrained_layout=True)
        for i, sid in enumerate(scenarios):
            for j, p in enumerate(PRIMARY):
                rr = perf[(perf.scenario_id == sid) & (perf.parameter == p)].set_index('method').reindex(METHODS)
                axes[i, j].bar(range(6), rr[metric], color=plt.cm.tab10(np.arange(6)))
                axes[i, j].set_title(f'{sid.replace("_late", "")} | {p} | n={int(rr.R_point.min())}')
                axes[i, j].set_xticks(range(6), [LABELS[m] for m in METHODS], rotation=60, ha='right')
                axes[i, j].set_ylim(bottom=0)
        fig.suptitle(title+' on identical six-method dataset IDs')
        save(fig, metric)
    for sid in scenarios:
        fig, axes = plt.subplots(2, 3, figsize=(10, 7), constrained_layout=True)
        for ax, p in zip(axes.flat, PRIMARY):
            rr = perf[(perf.scenario_id == sid) & (perf.parameter == p)].set_index('method').reindex(METHODS)
            ax.bar(range(6), rr.mse, color=plt.cm.tab10(np.arange(6)))
            ax.set_title(p); ax.set_ylabel('MSE')
            ax.set_xticks(range(6), [LABELS[m] for m in METHODS], rotation=55, ha='right')
        axes.flat[-1].axis('off')
        fig.suptitle(sid.replace('_late', '')+' | paired original datasets, n=20')
        save(fig, 'mse_'+sid)
    ip = pd.DataFrame(tables['interval_performance'])
    for metric, title in [('coverage_probability', 'Coverage probability'), ('average_width', 'Average interval width')]:
        fig, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
        for i, sid in enumerate(scenarios):
            for j, p in enumerate(PRIMARY):
                rr = ip[(ip.scenario_id == sid) & (ip.parameter == p)]
                rr = rr[((rr.method.str.startswith('mle')) & (rr.interval_family == 'log')) | (rr.interval_kind == 'credible')]
                rr = rr.set_index('method').reindex(METHODS)
                axes[i, j].bar(range(6), rr[metric], color=plt.cm.tab10(np.arange(6)))
                axes[i, j].set_title(f'{sid.replace("_late", "")} | {p}')
                axes[i, j].set_xticks(range(6), [LABELS[m] for m in METHODS], rotation=60, ha='right')
                if metric == 'coverage_probability':
                    axes[i, j].axhline(.95, color='black', ls='--', lw=1)
                    axes[i, j].set_ylim(0, 1.05)
        fig.suptitle(title+' | log-Wald confidence / shortest posterior credible intervals; missing bars unsupported')
        save(fig, metric)
    pairs = pd.DataFrame(tables['relative_mse_comparisons'])
    for name, contrasts in [('restriction_effects', [(METHODS[i], METHODS[i+1]) for i in (0, 2, 4)]),
                            ('hierarchical_vs_independent', [(METHODS[2], METHODS[4]), (METHODS[3], METHODS[5])])]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, sid in zip(axes, scenarios):
            for index, (u, v) in enumerate(contrasts):
                rr = pairs[(pairs.scenario_id == sid) & (pairs.method_U == u) & (pairs.method_R == v)].set_index('parameter').reindex(PRIMARY)
                ax.plot(PRIMARY, rr.relative_mse_reduction_percent, marker='o', label=f'{LABELS[u]} to {LABELS[v]}')
            ax.axhline(0, color='black', lw=1)
            ax.set_title(sid.replace('_late', '')); ax.set_ylabel('MSE reduction (%)'); ax.legend(fontsize=7)
        save(fig, name)
    d = pd.DataFrame(tables['bayesian_diagnostics'])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric, cutoff in zip(axes, ['rhat', 'ess_bulk', 'ess_tail'], [1.01, 400, 400]):
        data = [d[d.method == m][metric].dropna() for m in METHODS[2:]]
        ax.boxplot(data, tick_labels=[LABELS[m] for m in METHODS[2:]], showfliers=True)
        ax.axhline(cutoff, color='red', ls='--'); ax.set_title(metric)
        ax.tick_params(axis='x', rotation=30)
    save(fig, 'mcmc_diagnostics')
    raw = pd.DataFrame(tables['replication_estimates'])
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    for ax, p in zip(axes, PRIMARY):
        for m in METHODS:
            rr = raw[(raw.parameter == p) & (raw.method == m) & raw.valid_point]
            ax.scatter(rr.truth, rr.estimate, s=12, alpha=.5, label=LABELS[m])
        lim = ax.get_xlim(); ax.plot(lim, lim, 'k--', lw=1); ax.set_title(p); ax.set_xlabel('True'); ax.set_ylabel('Estimated')
    axes[-1].legend(fontsize=7)
    save(fig, 'estimated_vs_true')
    # Saved original rep 20 in each scenario: no extra fitting for illustrations.
    for sid in scenarios:
        for m in METHODS[2:]:
            rr = [r for r in tables['selected_fits'] if r['scenario_id'] == sid and r['method'] == m and r['repetition'] == 20]
            if not rr: continue
            b, _ = read_json(rr[0]['selected_path'])
            artifact = b.get('draw_artifact')
            if not artifact: continue
            with np.load(ROOT/artifact['path'], allow_pickle=False) as f:
                x, names = f['draws'], list(f['names'])
            fig, axes = plt.subplots(2, len(names), figsize=(3*len(names), 5), squeeze=False, constrained_layout=True)
            for j, name in enumerate(names):
                for chain in range(x.shape[0]):
                    axes[0, j].plot(x[chain, :, j], lw=.4, alpha=.65)
                    axes[1, j].hist(x[chain, :, j], bins=40, density=True, histtype='step')
                axes[0, j].set_title(name); axes[0, j].set_xlabel('Retained iteration')
                axes[1, j].set_xlabel(name); axes[1, j].set_ylabel('Posterior density')
            fig.suptitle(f'{sid} | {LABELS[m]} | original dataset rep 20 | four chains')
            save(fig, f'trace_posterior_{sid}_{m}')


def write_summary(dest, tables, validation):
    counts = pd.DataFrame(tables['valid_fit_counts'])
    perf = pd.DataFrame(tables['six_method_paired'])
    ip = pd.DataFrame(tables['interval_performance'])
    lines = ['# Verified local simulation completion', '',
        f"Selected valid fits: {validation['valid']}/360; missing: {validation['missing']}; invalid: {validation['invalid']}.", '',
        'Original 60 datasets and all historical records are preserved. Selection uses the historical valid result first, then the first passing effort level. '
        'The completion registry explicitly combines provenance across campaigns with identical priors/model/data but different implementation and sampling effort. '
        'No seed selection or threshold relaxation is used.', '',
        '| Method | Strict | Equality | Violation | Total |', '|---|---:|---:|---:|---:|']
    for m in METHODS:
        values = [int(counts[(counts.method == m) & (counts.scenario_id == s)].valid.iloc[0]) for s in counts.scenario_id.unique()]
        lines.append(f'| {LABELS[m]} | '+ ' | '.join(map(str, values+[sum(values)]))+' |')
    for sid in perf.scenario_id.unique():
        lines += ['', '## '+sid, '', '| Method | Parameter | n paired | AB | MSE |', '|---|---|---:|---:|---:|']
        for m in METHODS:
            for p in PRIMARY:
                r = perf[(perf.scenario_id == sid) & (perf.method == m) & (perf.parameter == p)].iloc[0]
                lines.append(f'| {LABELS[m]} | {p} | {r.R_point} | {r.absolute_bias:.8g} | {r.mse:.8g} |')
        lines += ['', '| Method | Parameter | Interval | n | CP | AW |', '|---|---|---|---:|---:|---:|']
        for r in tables['interval_performance']:
            if r['scenario_id'] == sid and r['parameter'] in PRIMARY and (r['interval_family'] in ('log', 'posterior_shortest')):
                cp = 'NA' if r['coverage_probability'] is None else f"{r['coverage_probability']:.6g}"
                aw = 'NA' if r['average_width'] is None else f"{r['average_width']:.8g}"
                lines.append(f"| {LABELS[r['method']]} | {r['parameter']} | {r['interval_kind']} | {r['R_CP']} | {cp} | {aw} |")
    lines += ['', '## Interpretation and limits', '',
        'The unrestricted hierarchy has lower observed MSE than unrestricted independent Bayes for every primary parameter in all three scenarios. '
        'Its shape MSE is 0.036386, 0.028812 and 0.046078 under strict ordering, equality and violation, versus MLE 0.051452, 0.037730 and 0.069376. '
        'At equality, restriction improves cause 1 and worsens cause 2 in both blocks for all three estimator families. '
        'Under false ordering, restricted Bayesian cause-2 coverage is 0.25--0.45; nominal posterior probability does not protect against a false model constraint.', '',
        'The order constraint leaves classical a and B unchanged. Under the independent scale prior, it also leaves their posterior distributions unchanged; '
        'separate method seeds introduce small Monte Carlo differences. This invariance does not hold for the hierarchy. '
        'Under equality, pooling and posterior restriction trade variance against asymmetric bias; inspect both causes and blocks. '
        'Under violation, any accuracy gain for one coefficient does not validate the false ordering: the other coefficient can deteriorate substantially.', '',
        'All ranks are descriptive, by scenario and parameter, on identical full-six dataset IDs. No aggregate ranking across differently scaled parameters is used. '
        'There are only 20 independent replications per scenario; CP changes in steps of 0.05, with binomial SE about 0.049 at CP=0.95. '
        'MSE difference Monte Carlo errors are supplied in relative_mse_comparisons.csv. '
        'Posterior mean MCSE is recorded parameter by parameter; ESS and Rhat do not prove convergence.', '',
        'Classical log-Wald intervals respect positivity. Raw Wald intervals are retained with their original labels and negative endpoints flagged in issues.csv. '
        'Restricted individual-b confidence intervals are unsupported and remain unavailable. Exact restricted probability sets can be empty; these count as zero coverage and zero width. '
        'Bayesian credible intervals are shortest contiguous sample intervals, not guaranteed disconnected HPD regions. '
        'Coverage is a frequentist simulation property even for credible intervals.', '',
        'Runtime summaries mix historical and optimized implementations and baseline/extended chain lengths, so they describe actual cost rather than a controlled sampler speed comparison. '
        'All-attempt totals include preserved failed fits. Posterior priors were retained without sensitivity selection; wider design and prior-sensitivity studies remain outside this 60-dataset completion.', '',
        '## Reproduction', '', 'From the repository root:', '', '```sh',
        '.venv/bin/python -m experiments.completion_study analyze',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study fit --workers 3', '```', '',
        'The first command rebuilds summaries/figures from saved results only. The second resumes only unresolved fits. '
        'Final results are under results/simulation_completion_v1/latest. Historical submission summaries remain unchanged.']
    (dest/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
