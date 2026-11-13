"""Independent pandas/numpy checks of the published saved-result analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .completion_study import OUT, ROOT, METHODS, EFFORT, acceptable
from .simulation_config import load_config, seed_for
from .simulation_storage import read_json
from src.bayesian_sampling import chain_diagnostics
from src.bayesian import posterior_reliability


def verify(dest):
    dest = Path(dest)
    points = pd.read_csv(dest/'replication_estimates.csv')
    intervals = pd.read_csv(dest/'replication_intervals.csv')
    perf = pd.read_csv(dest/'parameter_performance.csv')
    ip = pd.read_csv(dest/'interval_performance.csv')
    paired = pd.read_csv(dest/'six_method_paired.csv')
    contrasts = pd.read_csv(dest/'relative_mse_comparisons.csv')
    fits = pd.read_csv(dest/'selected_fits.csv')
    inv = pd.read_csv(dest/'inventory.csv')
    assert len(inv) == len(fits) == 360 and inv.dataset_id.nunique() == 60
    assert not fits.duplicated(['dataset_id', 'method']).any()
    assert not points.duplicated(['dataset_id', 'method', 'parameter']).any()
    assert not intervals.duplicated(['dataset_id', 'method', 'parameter', 'interval_family']).any()
    cfg = load_config('configs/simulation_submission.json')
    draw_diagnostic_checks = 0
    for r in fits.to_dict('records'):
        b, _ = read_json(r['selected_path'])
        assert acceptable(b)
        campaign, _ = read_json(ROOT/'metadata'/'fit_configs'/(b['fit_id']+'.json'))
        settings = campaign['settings']
        for key in ('hierarchical_prior', 'classical', 'mission_times', 'include_boundary_suprema', 'mcmc_master_seed'):
            assert settings[key] == cfg[key]
        if b['method'].startswith('bayes'):
            assert b['mcmc_seed'] == seed_for(cfg['mcmc_master_seed'], 'kw-bpcs-mcmc-v1', b['dataset_id'], b['method'])
            assert (b['mcmc']['warmup'], b['mcmc']['draws']) in EFFORT
            for key in ('chains', 'max_rhat', 'min_ess', 'probability', 'slice_width', 'max_slice_steps'):
                assert b['mcmc'][key] == settings['mcmc'][key] == cfg['mcmc'][key]
        artifact = b.get('draw_artifact')
        if artifact:
            with np.load(ROOT/artifact['path'], allow_pickle=False) as f:
                x, names = f['draws'], tuple(f['names'])
            checks = {name: x[:, :, j] for j, name in enumerate(names)}
            reliability = posterior_reliability(x, names, ('A', 'B'), cfg['mission_times'])
            checks.update({label+'/'+name: values for label, entries in reliability.items() for name, values in entries.items()})
            for d in b['bayesian_diagnostics']:
                if d['parameter'] not in checks:
                    continue
                actual = chain_diagnostics(checks[d['parameter']])
                for key in ('rhat', 'ess_bulk', 'ess_tail', 'ess_mean', 'mcse_mean'):
                    assert np.isclose(actual[key], d[key], rtol=1e-12, atol=1e-12)
                draw_diagnostic_checks += 1
    valid = points[points.valid_point]
    assert np.isfinite(valid.estimate).all() and (valid.estimate > 0).all()
    for r in valid.to_dict('records'):
        name = r['parameter']
        if name.startswith('p[') or 'probability' in name or 'incidence' in name or 'survival' in name or 'median' in name:
            assert r['estimate'] <= 1
        if r['method'].endswith('within_block') and (name.startswith('p[') or name.endswith('cause_1_probability')):
            assert r['estimate'] <= .5
    keys = ['scenario_id', 'method', 'parameter']
    grouped = {k: g for k, g in points.groupby(keys)}
    for r in perf.to_dict('records'):
        g = grouped[tuple(r[k] for k in keys)]
        g = g[g.valid_point]
        errors = g.estimate.to_numpy()-g.truth.to_numpy()
        assert r['R_point'] == len(g)
        assert np.isclose(r['absolute_bias'], abs(errors.mean()), rtol=1e-12, atol=1e-13)
        assert np.isclose(r['mse'], np.square(errors).mean(), rtol=1e-12, atol=1e-13)
    for r in paired.to_dict('records'):
        ids = json.loads(r['dataset_ids'])
        assert len(ids) == r['R_point'] == 20
        for m in METHODS:
            g = grouped[(r['scenario_id'], m, r['parameter'])]
            assert set(ids) == set(g[g.valid_point].dataset_id)
        g = grouped[tuple(r[k] for k in keys)].set_index('dataset_id').loc[ids]
        assert np.isclose(r['mse'], ((g.estimate-g.truth)**2).mean(), rtol=1e-12, atol=1e-13)
    ikeys = keys+['interval_family', 'interval_kind']
    igrouped = {k: g for k, g in intervals.groupby(ikeys)}
    for r in ip.to_dict('records'):
        g = igrouped[tuple(r[k] for k in ikeys)]; g = g[g.valid_interval]
        assert len(g) == r['R_CP'] == r['R_AW']
        if len(g):
            empty = g.status == 'empty'
            normal = g[~empty]
            assert np.isfinite(normal.lower).all() and np.isfinite(normal.upper).all()
            assert (normal.lower <= normal.upper).all()
            assert ((normal.lower <= normal.truth) & (normal.truth <= normal.upper)).astype(int).eq(normal.covered).all()
            assert np.isclose(r['coverage_probability'], g.covered.mean())
            assert np.isclose(r['average_width'], g.width.mean())
            if r['interval_family'] == 'log':
                assert (normal.lower > 0).all()
    for r in contrasts.to_dict('records'):
        ids = json.loads(r['dataset_ids'])
        assert len(ids) == r['R_paired'] == 20
        u = grouped[(r['scenario_id'], r['method_U'], r['parameter'])].set_index('dataset_id').loc[ids]
        v = grouped[(r['scenario_id'], r['method_R'], r['parameter'])].set_index('dataset_id').loc[ids]
        assert u.dataset_sha256.equals(v.dataset_sha256)
        eu, ev = (u.estimate-u.truth)**2, (v.estimate-v.truth)**2
        assert np.isclose(r['MSE_U_paired'], eu.mean(), atol=1e-13)
        assert np.isclose(r['MSE_R_paired'], ev.mean(), atol=1e-13)
        assert np.isclose(r['paired_difference_mcse'], (eu-ev).std(ddof=1)/np.sqrt(len(ids)), atol=1e-13)
    tex = (dest/'manuscript_simulation.tex').read_text()
    for r in paired[paired.parameter.isin(['a', 'b1[A]', 'b2[A]', 'b1[B]', 'b2[B]'])].to_dict('records'):
        assert f"{r['mse']:.6f}" in tex and f"{r['absolute_bias']:.6f}" in tex
    result = dict(status='passed', point_groups=len(perf), interval_groups=len(ip),
                  six_method_groups=len(paired), paired_contrasts=len(contrasts),
                  selected_fits=len(fits), datasets=60, all_six_dataset_denominator=20,
                  recomputed_parameterwise_draw_diagnostics=draw_diagnostic_checks,
                  publication_primary_values_match_csv=True)
    (dest/'independent_qc.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(OUT/'latest'), indent=2))
