# Verified simulation completion, 13 November 2026

The original submission study is complete: **360/360 valid selected fits**, on
the original 60 saved datasets. Each of the six methods has 60 valid fits and
uses the same 20 original dataset IDs within each scenario. Nothing was pushed.
All sampling, analysis, testing and PDF compilation ran on this local MacBook.
The LaTeX compiler downloaded missing package/font resources; no computation
was sent to a remote service.

## Design, model and priors

The executed configuration is `configs/simulation_submission.json`, not the
600-dataset pilot or 24,000-dataset main configuration. Both blocks have n=50,
m=30, common a=1.5, and withdrawals (0,...,0,20). Strict, equality and violation
parameters are respectively A=(.5,1.5)/(1,1)/(1.5,.5),
B=(.75,2.25)/(1.5,1.5)/(2.25,.75). Totals remain 2 and 3.
The master generation seed is 20261101; MCMC master seed is 20261102.
`datasets.csv` records all original dataset IDs, exact seed integers and hashes.
Large seed integers must be read as strings or Python integers, not float64.

The independent latent cause model yields minimum survival
(1-t^a)^(b1+b2) and subdensity a*b_c*t^(a-1)*(1-t^a)^(b1+b2-1).
Progressive withdrawals give exponent (1+r)*(b1+b2)-1, hence the likelihood
with W=-sum((1+r)*log(1-x^a)). The verified classical conditional estimates are
b_c=d_c/W, B=m/W and restricted p=min(d1/m,.5). Restricted shape and totals
are unchanged. Zero cause counts are boundary cases, not positive interior
estimates; none prevents an attained MLE in this particular saved study.

The independent prior remains proportional to 1/(a*product(b_ic)). Its
conditional totals are Gamma(m,W), with independent Beta(d1,d2) proportions,
truncated at .5 when restricted. This truncation leaves the a/total posterior
distributions unchanged. Separate original method seeds cause small Monte Carlo
differences in their estimated summaries.

The hierarchy remains a~Gamma(2,2), kappa_c~Gamma(3,2),
lambda_c~Gamma(2,2), b_ic|kappa,lambda~Gamma(kappa_c,lambda_c), in shape-rate
notation. All parameter-dependent Gamma normalizers are retained. The restricted
hierarchy conditions the *entire joint posterior* on the order event; it does
not introduce hyperparameter-dependent event-normalization factors. Its
coordinate truncated-Gamma, Gamma and log-slice conditionals are unchanged.
See `docs/BAYESIAN_DERIVATIONS.md` for the full formulas and moment conditions.

No statistical formula or prior defect was found. The only estimator change is
an equivalent evaluation of the hierarchical a-conditional from sufficient
statistics, avoiding the repeated per-observation Python likelihood loop.
Regression tests compare conditional log-density differences against the
independent row-wise likelihood across four widely separated shape pairs.
Existing moment, propriety, boundary and convergence tests remain intact.

## Actual unfinished work and execution

The inventory confirmed the historical success counts 60/60/31/31/31/27.
There were 109 missing combinations and 11 diagnostic failures, all of the
latter restricted hierarchical fits in the violation scenario. Their primary
bulk ESS could be as low as 102.76 and R-hat as high as 1.06137. Interrupted
task records also document prior budget limits; process completion alone was
never sufficient for validity.

All 240 original successful fit records were reused. The completion added 120
successful fits and preserved eight further baseline diagnostic failures (seven
restricted hierarchical and one unrestricted hierarchical) before longer retries.
Thus all 360 intended combinations were attempted and evaluated. All 19 failed
attempts remain accessible across the historical and completion directories.

The machine reports 10 CPU cores and 16 GiB RAM, Python 3.12.14, macOS ARM64.
Three spawned local worker processes and one BLAS/OpenMP thread per process
bounded parallelism. New fits consumed 2,371.7326 summed process-seconds; this
is not elapsed wall time. An exact outer wall-time log was not captured for
the first completion invocation; subsequent resume invocations record it.
Historical fit runtimes mix implementations and machine conditions and should
not be interpreted as a controlled speed benchmark.

The predeclared effort levels, per chain, are (warmup, retained):
(1000,2000), (2000,8000), (4000,32000), always four chains. Known historical
diagnostic failures enter at level 1. New missing fits start at level 0.
All completed by level 1: 18 restricted hierarchical and one unrestricted
hierarchical success required that longer level. No level-2 run was necessary.
Seeds remain the original deterministic method/dataset seeds at every level;
there is no search for favorable seeds. Failed attempts are never overwritten.

Every selected Bayesian fit passes R-hat <=1.01 and bulk/tail ESS >=400 for
every monitored parameter, hyperparameter and requested reliability transform.
The maximum selected R-hat is 1.00952896; minimum bulk ESS 419.64408;
minimum tail ESS 865.85960. Primary first and second moments are established.
The maximum posterior-mean MCSE/absolute mean among primary coefficients is
about 1.47%; small tail survival probabilities have ratios up to 4.03%.
These are reported precision measures, not new acceptance thresholds.
Passing diagnostics remains a necessary check, not proof of convergence.

Saved draws from original repetition 20 in every scenario supply representative
traces and posterior histograms for all four Bayesian methods, without extra
refitting. The audit recomputes 444 parameter-wise diagnostics from these draws
and matches the stored values. Historical successful fits without saved draws
are validated from their hash-verified stored parameter-wise diagnostics and
moment records; no claim is made that their unsaved traces were reexamined.

## Results and fair comparisons

All six methods now use identical sets of 20 original IDs in each scenario.
`six_method_paired.csv` gives all 702 parameter/reliability groups. The 1,755
pairwise contrasts (15 method pairs x 39 targets x 3 scenarios) also use the
six-way common sets, with paired squared-error-difference Monte Carlo SEs.
Matching counts without matching IDs is explicitly prohibited by the regression
test and independent audit.

Shape MSE on these matched sets is:

| Method | Strict | Equality | Violation |
|---|---:|---:|---:|
| MLE unrestricted | .051452 | .037730 | .069376 |
| MLE restricted | .051452 | .037730 | .069376 |
| Independent unrestricted | .049317 | .037176 | .067366 |
| Independent restricted | .049215 | .037293 | .067298 |
| Hierarchical unrestricted | .036386 | .028812 | .046078 |
| Hierarchical restricted | .036750 | .028690 | .043810 |

The unrestricted hierarchy has lower observed MSE than unrestricted independent
Bayes for every primary coefficient in all three scenarios. Against MLE, its
shape squared-error reductions have paired Monte Carlo SEs .004991, .003041
and .009776 for observed differences .015066, .008918 and .023298. With only
20 replications and many comparisons, these are descriptive design-specific
results, not a universal ranking or a multiple-testing-adjusted claim.

At strict ordering no classical pooling occurs in either block. Bayesian
restriction has small mixed effects. At equality, all three restriction
comparisons improve cause 1 and worsen cause 2 in both blocks. Under violation,
classical block-A cause-1 MSE falls 1.000758 -> .396180 but cause-2 MSE rises
.064089 -> .829559. Both Bayesian families show the same directional tradeoff,
including block B. An apparent gain in one coefficient cannot validate a false
ordering assumption.

Restricted Bayesian cause-2 interval coverage under violation is .35/.45 for
independent A/B and .25/.45 for hierarchical A/B. Narrow conditional credible
intervals can be unreliable under a false restriction. Primary interval tables
retain distinct classical confidence and posterior credible labels and all
denominators. AB is absolute mean error, not mean absolute error.

The manuscript emphasizes positive log-Wald classical intervals. The original
raw Wald procedures remain in the full export: 11 negative endpoint records
are flagged in `issues.csv`, not clipped, discarded or relabeled as positive
intervals. Restricted individual-b classical intervals remain unsupported;
their denominator is zero, with no fabricated endpoints. Bayesian B-total
intervals remain unsupported by the existing adapter. Exact restricted
probability sets can be empty; they count as zero coverage and zero set length.
With only 20 intervals, CP moves in .05 increments, and its nominal binomial
MCSE is approximately .049. Wider replication/design and prior sensitivity
remain important scientific limitations, not unfinished fits in this study.

## Reproduction and resumption

From `/Users/mac_m5/Desktop/Personal/Stat's Project`:

```sh
MPLCONFIGDIR=tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study analyze
```

This original end-to-end command reads saved experimental records, rebuilds the
analysis and additionally invokes the PDF compiler. Current saved-result-only
reproduction, which does not need TeX, is documented in the root
[README](../README.md). The later-edited manuscript requires TeX packages that
are not all in the current local cache; the statistical analysis does not depend
on them. No manuscript recompilation is needed to inspect or verify the results.

Resume unfinished fitting with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.completion_study fit --workers 3
```

It currently schedules zero new fits. Atomic immutable JSON commits, dataset
hash validation and task locks make it resumable. If interrupted workers have
actually stopped, append `--recover-locks`; recovery refuses live local owners.
No successful fit is rerun, and generation is not part of this workflow.

Other checks:

```sh
.venv/bin/python -m experiments.completion_study inventory
.venv/bin/python -m experiments.completion_qc
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -W error
git diff --check
```

## Exact local deliverables

The base directory is
`/Users/mac_m5/Desktop/Personal/Stat's Project/results/simulation_completion_v1/latest/`.
`latest` is a symlink to the final immutable-input analysis snapshot. Every file
below is beneath that exact directory. `artifact_manifest.json` provides exact
resolved paths, SHA-256 hashes and sizes for the generated analysis artifacts.
Its root-report entries refer to the original completion build: the manuscript
was subsequently edited, and the broken root PDF was removed at the user's
request. The manifest is preserved as historical provenance, not rewritten to
claim a new report verification.

| File | Contents |
|---|---|
| `SUMMARY.md` | All primary scenario/method AB, MSE, interval CP/AW and interpretation |
| `datasets.csv` | All 60 original IDs, data hashes and random-seed provenance |
| `inventory.csv` | All 360 combinations and their selected/attempt paths |
| `selected_fits.csv` | 360 selected records with campaign and record hashes |
| `attempt_history.csv` | Preserved historical and new successful/failed attempts |
| `valid_fit_counts.csv` | Counts by scenario and estimator |
| `replication_estimates.csv` | Every raw point estimate and validity/moment status |
| `replication_intervals.csv` | Every interval, validity, endpoints, coverage and width |
| `parameter_performance.csv` | All-target AB/MSE with denominators |
| `interval_performance.csv` | CP/AW by interval procedure, with both denominators |
| `six_method_paired.csv` | Full six-way matched parameter performance |
| `paired_dataset_ids.csv` | Explicit actual common IDs per scenario/target |
| `relative_mse_comparisons.csv` | All 15 contrasts per scenario/target and paired MCSE |
| `rankings.csv` | Descriptive within-scenario, within-parameter MSE ranks |
| `runtime.csv`, `runtime_summary.csv` | Selected and all-attempt costs |
| `bayesian_diagnostics.csv` | All selected parameter-wise R-hat, ESS, MCSE |
| `mcmc_summary.csv`, `mcse_summary.csv` | Diagnostic/Monte Carlo precision summaries |
| `issues.csv` | Eleven negative raw-Wald endpoint flags |
| `validation.json` | Data/fit/diagnostic/interval integrity and completeness |
| `independent_qc.json` | Independently recomputed metric/paired/draw checks |
| `manuscript_simulation.tex` | Tables and discussion generated from the same rows as CSV |
| `artifact_manifest.json` | Full artifact path/hash/size index |

`markdown/` contains nine named table companions: `valid_fit_counts.md`,
`parameter_performance.md`, `interval_performance.md`, `six_method_paired.md`,
`relative_mse_comparisons.md`, `rankings.md`, `runtime_summary.md`,
`mcmc_summary.md`, and `mcse_summary.md`. CSV retains full round-trip precision;
Markdown uses 12 significant digits and manuscript tables six decimals.

`figures/` contains each following stem in **both `.png` and `.pdf`**:
`mse`, `absolute_bias`, `coverage_probability`, `average_width`,
`restriction_effects`, `hierarchical_vs_independent`, `mcmc_diagnostics`,
`estimated_vs_true`, `mse_strict_order_late`, `mse_equality_late`,
`mse_violation_late`. There are also twelve exact stems
`trace_posterior_<scenario>_<method>` for each scenario
`strict_order_late`, `equality_late`, `violation_late` and each method
`bayes_independent_unrestricted`, `bayes_independent_within_block`,
`bayes_hierarchical_unrestricted`, `bayes_hierarchical_within_block`.
Each trace figure also contains marginal posterior histograms for all fitted
parameters, including hyperparameters for the hierarchy. The manifest enumerates
all 46 figure paths individually.

Additional exact repository-root-relative paths:

- `Report.tex`: current manuscript source, preserved unchanged during cleanup.
- `output/pdf/Report.pdf`: valid saved 18-page completion build. The broken
  later root PDF was removed as requested; no replacement was compiled.
- `output/pdf/Report_pre_completion.pdf`: preserved previous top-level PDF.
- `output/pdf/Report_output_pre_completion.pdf`: preserved previous output PDF.
- `output/pdf/compilation.txt`: local compiler output.
- `results/simulation_completion_v1/attempts/`: immutable new fit records.
- `results/simulation_submission/draws/completion_v1/`: compressed draw checkpoints.
- `results/simulation_completion_v1/audits/`: initial inventory and preservation audit.
- `results/simulation_completion_v1/fitting.log`: actual fit outcomes and runtimes.

Original submission CSV pointers/manifests and successful records remain
unchanged; they describe the historical partial campaign. The completion
registry deliberately combines those preserved valid fits with separately
identified new campaigns having identical data, model, priors and thresholds.
Source implementations and chain effort differ and remain explicit. Do not
mistake the old submission manifest's 31/31/31/27 counts for the final registry.
Real-data estimates and the manuscript's entire real-data section remain
byte-for-byte unchanged.
