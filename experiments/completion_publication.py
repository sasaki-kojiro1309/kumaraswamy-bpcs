"""Generate manuscript tables directly from the verified CSV-equivalent rows."""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
import subprocess
import hashlib
import json

from .completion_analysis import METHODS, LABELS, PRIMARY

TEX_PARAMS = [r'$a$', r'$b_{A1}$', r'$b_{A2}$', r'$b_{B1}$', r'$b_{B2}$']


def compile_report():
    """Compile on this device using cached TeX resources; no remote execution."""
    binary = os.environ.get('TECTONIC') or shutil.which('tectonic')
    bundled = Path('/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic')
    if not binary and bundled.is_file():
        binary = str(bundled)
    if not binary:
        raise RuntimeError('Analysis is saved; set TECTONIC to a local Tectonic executable to compile the PDF.')
    output = Path('output/pdf')
    output.mkdir(parents=True, exist_ok=True)
    for src, backup in [('Report.pdf', 'Report_pre_completion.pdf'),
                        ('output/pdf/Report.pdf', 'Report_output_pre_completion.pdf')]:
        if Path(src).exists() and not (output/backup).exists():
            shutil.copy2(src, output/backup)
    result = subprocess.run([binary, '-C', '--keep-logs', '--outdir', str(output), 'Report.tex'],
                            capture_output=True, text=True)
    (output/'compilation.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Local LaTeX compilation failed; inspect output/pdf/compilation.txt. Saved analysis is intact.')
    shutil.copy2(output/'Report.pdf', 'Report.pdf')
    latest = Path('results/simulation_completion_v1/latest').resolve()
    files = list(latest.rglob('*'))+[Path('Report.tex'), Path('Report.pdf')]
    manifest = {str(p.relative_to(Path.cwd()) if p.is_absolute() else p):
                dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                for p in files if p.is_file() and p.name != 'artifact_manifest.json'}
    (latest/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')


def write_manuscript(dest, tables, validation):
    perf = pd.DataFrame(tables['six_method_paired'])
    ip = pd.DataFrame(tables['interval_performance'])
    counts = pd.DataFrame(tables['valid_fit_counts'])
    scenarios = ['strict_order_late', 'equality_late', 'violation_late']
    snames = ['Strict ordering', 'Equality', 'Ordering violation']
    lines = [r'\subsubsection*{Completed local execution and validation}',
        f"All {validation['original_datasets']} original datasets were retained with their recorded identities and seeds. "
        f"The final registry has {validation['valid']} valid fits out of 360 planned combinations, "
        f"with {validation['missing']} missing and {validation['invalid']} unresolved invalid combinations. "
        'All computation was executed locally on the MacBook (10 CPU cores, 16 GiB RAM), with three worker processes and one BLAS/OpenMP thread per worker.',
        r'The baseline MCMC uses four independent chains, 1,000 warm-up and 2,000 retained iterations per chain. '
        r'Failed diagnostics trigger a predetermined increase to 2,000/8,000 warm-up/retained iterations, then 4,000/32,000 if needed. '
        r'The original method--dataset seed, priors, likelihood and thresholds ($\widehat R\le1.01$, bulk and tail ESS $\ge400$ for every monitored parameter and reliability quantity) are retained. '
        'Historical successes are reused without refitting. Historical failures remain available. The first acceptable effort level is selected; no favorable-seed selection is performed.',
        'The historical counts were 60, 60, 31, 31, 31 and 27. There were 109 uncompleted combinations and 11 restricted hierarchical diagnostic failures. '
        'The completion computed 120 successful replacements/additions and preserved eight additional baseline diagnostic failures before longer retries. '
        'All 19 historical/new failed records are retained. All additional successful fits passed by the second effort level; the third level was unnecessary.',
        r'The hierarchical shape update now evaluates the identical conditional kernel '
        r'$(M+A_a-1)\log a+(a-1)T-B_a a-\sum_iH_i(a)-\sum_iB_iW_i(a)$ '
        'from validated block arrays rather than rebuilding each observation contribution at every slice proposal. '
        'Independent row-likelihood ratio regression checks verify this optimization. The posterior target and Gibbs/slice kernels are unchanged.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Verified valid fits. Every method uses the same original 20 datasets within each scenario.}\label{tab:sim-counts}',
        r'\begin{tabular}{lrrrr}\hline Method & Strict & Equality & Violation & Total\\\hline']
    for m in METHODS:
        n = [int(counts[(counts.method == m) & (counts.scenario_id == sid)].valid.iloc[0]) for sid in scenarios]
        lines.append(LABELS[m]+' & '+' & '.join(map(str, n+[sum(n)]))+r'\\')
    lines += [r'\hline\end{tabular}\end{table}',
        r'\subsubsection*{Fair six-method parameter accuracy}',
        'The six-way comparison takes the intersection of actual original dataset IDs and verifies their data hashes and truths. '
        'It contains all 20 datasets in every scenario for all five primary parameters. '
        'Tables below report AB and MSE for both causes in both blocks; no ranking is based on incompatible subsets. '
        'Full precision estimates, auxiliary quantities, 15 pairwise contrasts, paired squared-error-difference Monte Carlo SEs, and dataset memberships are saved in the versioned CSV exports.']
    for sid, title in zip(scenarios, snames):
        for metric, label in [('absolute_bias', 'Absolute bias'), ('mse', 'Mean squared error')]:
            lines += [r'\begin{table}[htbp]\centering\small',
                r'\caption{'+title+': '+label.lower()+r' on the identical 20 datasets. All denominators are 20.}',
                r'\begin{tabular}{lrrrrr}\hline Method & '+' & '.join(TEX_PARAMS)+r'\\\hline']
            for m in METHODS:
                rr = perf[(perf.scenario_id == sid) & (perf.method == m)].set_index('parameter')
                lines.append(LABELS[m]+' & '+' & '.join(f'{rr.loc[p, metric]:.6f}' for p in PRIMARY)+r'\\')
            lines += [r'\hline\end{tabular}\end{table}']
        def mse(m, p):
            return float(perf[(perf.scenario_id == sid) & (perf.method == m) & (perf.parameter == p)].mse.iloc[0])
        u, h = mse(METHODS[0], 'a'), mse(METHODS[4], 'a')
        if sid == 'strict_order_late':
            lines += [r'\paragraph{Strict ordering.} '
                'Neither block activates classical pooling, so both MLEs coincide. '
                f'Hierarchical unrestricted shape MSE is {h:.6f}, versus {u:.6f} for MLE '
                f'({100*(u-h)/u:.1f}'+r'\% lower in this experiment). '
                'The hierarchical prior also lowers the observed MSE of both cause coefficients in block B. '
                'The independent Bayesian procedure does not uniformly improve cause-parameter MSE over MLE. '
                'Restriction changes Bayesian cause allocation slightly, with mixed effects across coefficients.']
        elif sid == 'equality_late':
            lines += [r'\paragraph{Equality.} '
                f'Restricted MLE reduces block-A cause-1 MSE from {mse(METHODS[0], "b1[A]"):.6f} to {mse(METHODS[1], "b1[A]"):.6f}, '
                f'while cause-2 MSE rises from {mse(METHODS[0], "b2[A]"):.6f} to {mse(METHODS[1], "b2[A]"):.6f}. '
                'The corresponding Bayesian restrictions also improve cause 1 and worsen cause 2 in both blocks. '
                'Pooling and one-sided posterior truncation therefore trade variance reduction against asymmetric estimation bias even at a correct equality boundary.']
        else:
            lines += [r'\paragraph{Ordering violation.} '
                f'Restricted MLE changes block-A cause-1 MSE from {mse(METHODS[0], "b1[A]"):.6f} to {mse(METHODS[1], "b1[A]"):.6f}, '
                f'but cause-2 MSE rises from {mse(METHODS[0], "b2[A]"):.6f} to {mse(METHODS[1], "b2[A]"):.6f}. '
                'Both Bayesian families show the same direction of tradeoff, including block B. '
                'A lower MSE for the larger true coefficient does not validate the false restriction; bias and variance can offset for one coordinate while worsening the other. '
                'The complete data-ID comparison resolves the earlier one-dataset restricted-hierarchical subset limitation.']
    lines += [r'\subsubsection*{Interval coverage and precision}',
        r'Classical entries below are positive log-Wald confidence intervals; Bayesian entries are 95\% shortest contiguous posterior-draw credible intervals. '
        'These are distinct procedures. Restricted individual-cause classical intervals remain unsupported and are not replaced by ordinary Wald intervals. '
        'All supported primary intervals have 20 valid replications per scenario. '
        r'Coverage changes in steps of 0.05; at nominal 0.95 its binomial Monte Carlo SE is about 0.049. '
        'Narrower intervals cannot be interpreted independently of coverage.']
    for sid, title in zip(scenarios, snames):
        lines += [r'\begin{table}[htbp]\centering\scriptsize',
            r'\caption{'+title+r': CP / AW for marginal 95\% intervals. Each available entry has denominator 20; -- denotes unsupported.}',
            r'\begin{tabular}{lrrrrr}\hline Method & '+' & '.join(TEX_PARAMS)+r'\\\hline']
        for m in METHODS:
            rr = ip[(ip.scenario_id == sid) & (ip.method == m)]
            rr = rr[rr.interval_family == ('log' if m.startswith('mle') else 'posterior_shortest')].set_index('parameter')
            values = []
            for p in PRIMARY:
                if p not in rr.index or not rr.loc[p, 'R_CP']:
                    values.append('--')
                else:
                    values.append(f"{rr.loc[p, 'coverage_probability']:.2f} / {rr.loc[p, 'average_width']:.3f}")
            lines.append(LABELS[m]+' & '+' & '.join(values)+r'\\')
        lines += [r'\hline\end{tabular}\end{table}']
    lines += [r'Under the deliberately false restriction, Bayesian credible intervals for the smaller true cause-2 coefficients can have very poor repeated-sampling coverage. '
        r'An interval is conditional on the assumed restricted model; a nominal 95\% posterior probability does not protect against a scientifically false constraint. '
        'The exact restricted probability confidence-set intersections can be empty: those are retained as zero coverage and zero set length. '
        'All interval families, explicit denominators and negative raw-Wald endpoint flags remain in the supplementary CSV files.',
        r'\subsubsection*{MCMC and computational performance}',
        'The historical restricted hierarchical failures were concentrated in order violation, with low bulk ESS and sometimes elevated split rank R-hat. '
        'Coordinate-wise truncated Gamma updates near the equality boundary mix more slowly in these cases. '
        'Longer fixed-seed runs resolved them without changing the ordering implementation. '
        'All monitored model parameters, hyperparameters and reliability transforms in every selected Bayesian fit meet the original criteria. '
        'Parameter-wise R-hat, bulk/tail ESS and posterior-mean MCSE are exported; representative traces and posterior densities use original repetition 20 in every scenario.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Observed local fitting cost in seconds. Selected means combine historical and optimized code and different chain lengths; they are not controlled speed benchmarks. All-attempt totals include failures.}',
        r'\begin{tabular}{lrrr}\hline Method & Mean selected & Median selected & All-attempt total\\\hline']
    for r in tables['runtime_summary']:
        lines.append(f"{LABELS[r['method']]} & {r['mean_selected_seconds']:.4f} & {r['median_selected_seconds']:.4f} & {r['total_all_attempt_seconds']:.3f}"+r'\\')
    lines += [r'\hline\end{tabular}\end{table}',
        'Runtime totals omit file I/O and orchestration; no precise total wall-clock runtime was captured for the initial completion invocation. '
        'The additional 128 fit attempts consumed about 2,371.7 summed process-seconds across the three workers. '
        'The actual posterior procedures remain substantially more expensive than the analytic/profile MLEs.',
        r'\subsubsection*{Interpretation, reproducibility and limitations}',
        'The hierarchy has lower observed shape MSE than MLE and the independent prior in all three scenarios, '
        'and its unrestricted estimator has lower MSE for every primary parameter than the independent unrestricted estimator in this study. '
        'This is evidence for these particular hyperpriors and this small design, not a universal Bayesian advantage. '
        'Independent scale-prior Bayesian estimates do not uniformly improve on MLE. '
        'Restriction leaves classical shape and totals unchanged; under the independent prior their posterior distributions are also invariant, '
        'up to Monte Carlo error from separate method seeds. The hierarchy does not share that invariance.',
        'All primary accuracy comparisons now use identical full-six data-ID sets. '
        'No ranking averages errors across parameters with different scales. '
        'Twenty replications per scenario, one sample size, and terminal withdrawals only remain major limits. '
        'Prior sensitivity, alternate censoring designs and much larger replication counts are needed before broad claims about estimator or interval superiority. '
        'Passing diagnostics is not a proof of convergence, and Monte Carlo error remains nonzero.',
        r'\noindent The saved-result analysis is reproduced locally by:',
        r'\begin{quote}\small\ttfamily .venv/bin/python -m experiments.completion\_study analyze\end{quote}',
        r'The fitting-resume command, dataset-ID registry, full precision tables and figure index are documented in '
        r'\texttt{docs/SIMULATION\_COMPLETION.md}. Historical results are preserved separately.']
    for sid, title in zip(scenarios, snames):
        lines += [r'\begin{figure}[p]\centering',
            r'\includegraphics[width=0.98\textwidth]{results/simulation_completion_v1/latest/figures/mse_'+sid+r'.pdf}',
            r'\caption{'+title+r': primary-parameter MSE for all six methods on the same 20 original datasets. U and R denote unrestricted and restricted. Axes start at zero; scales differ by parameter.}',
            r'\end{figure}']
    lines.append(r'\clearpage')
    (dest/'manuscript_simulation.tex').write_text('\n\n'.join(lines)+'\n')
