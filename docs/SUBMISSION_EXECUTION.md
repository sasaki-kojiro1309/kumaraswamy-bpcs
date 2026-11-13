# Submission computation, 1 November 2026

Historical execution record. The 13 November completion has 360 valid fits;
see [Simulation completion](SIMULATION_COMPLETION.md). Original counts and
submission pointers in this dated record remain preserved.

This execution uses `configs/simulation_submission.json`, not the original pilot.
There are exactly 60 saved datasets: 20 each for strict ordering, equality, and
ordering violation, with two independent blocks, n=50, m=30, a=1.5, and
withdrawals `(0,...,0,20)`. Original configurations and earlier results remain.

The authoritative current execution audit is
`results/simulation_submission/manifest.json`. It points to an immutable audit
with exact counts, fit timings, data/result integrity checks, and the analysis
version. CSV paths under `estimates`, `diagnostics`, and `summaries` point to
versioned exports. Missing method fits are explicitly `not_run`; failed MCMC
diagnostics are retained and excluded from valid estimates and intervals.

## Execution and budgets

All 60 datasets were saved before fitting. Both classical methods achieved 60
interior MLEs each. Their recorded individual fit durations total approximately
1.226 seconds; this sum excludes serialization and verification overhead. The
initial MLE phase did not record a separate outer wall-clock duration. A later
resume invocation reused all 120 committed results and completed no new fits.

The initial one-dataset Bayesian benchmark used all four unchanged methods:
independent unrestricted 8.023 s, independent restricted 8.168 s, hierarchical
unrestricted 72.120 s, and hierarchical restricted 75.565 s. The sum is 163.875 s.
For 60 datasets that implies about 164 minutes serial, or 55 minutes even under
ideal three-worker scaling, before contention and diagnostics failures. Full
Bayesian coverage is therefore not promised within 15 minutes.

The resumed simulation uses three process workers, one per scenario, and an
absolute deadline of 2026-11-01 15:25:45 UTC. The supervisor's started/finished
logs record observed wall time. POSIX elapsed-time alarms stop in-flight fits
five seconds before the shared deadline; a parent deadline additionally stops
remaining workers. Interrupted tasks receive separate diagnostic records and
are eligible for resumption. Every completed fit is atomically committed before
the next task starts. A prior run was stopped at the user's instruction; its
completed fits are reused and its missing completion log is identified openly.

Successful committed task files are never overwritten. An identical invocation
with the same config, implementation and a new future deadline skips them.
Committed diagnostic failures are also preserved; retrying them requires an
explicit separate campaign attempt. Do not recover locks while workers run.

Example resume, with a newly calculated Unix deadline:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.submission_supervisor --config configs/simulation_submission.json --deadline-unix NEW_FUTURE_UNIX_TIME --workers-per-scenario 1
```

After the workers stop, regenerate audits and summaries without fitting:

```sh
.venv/bin/python -m experiments.submission_finalize
```

Four chains, 1000 warmup iterations and 2000 retained draws per chain are
unchanged. R-hat <=1.01, bulk/tail ESS >=400, posterior propriety and moment
checks remain in force. Passing numerical diagnostics is not proof of
convergence. The same saved observations and dataset hash identify every fit.

## Performance summaries

`summaries/parameter_performance.csv` contains mean estimates, signed bias,
absolute bias, MSE, standard deviation and valid repetition counts.
`summaries/interval_performance.csv` contains CP, AW and separate interval
denominators. Unsupported restricted classical intervals remain unavailable.
`summaries/relative_mse_comparison.csv` uses common successful datasets.
Success rates, MCMC summaries and active pooling are separate CSVs.

AB is the absolute mean error, not mean absolute error. Conditional metrics
expose valid counts; failures and missing fits remain visible in denominators
and logs. Initial exports were independently recomputed for 702 point-metric
groups and 756 interval-metric groups. A final audit verifies the saved designs,
data/result hashes and unchanged report/PDF hashes.

Twenty planned repetitions per scenario is a small exploratory study. Bayesian
subsets are smaller still. Small MSE differences do not establish superiority;
credible intervals do not guarantee nominal frequentist coverage. Order-violating
restricted comparisons are explicitly marked as model misspecification.

## Real-data application

`results/real_data/validation/completed_validation_summary.json` is the current
real-data outcome. The initial Hoel and melanoma screening records are retained
as rejected-candidate history, not as the final selected application.

Hoel has 181 mice in two actual experimental groups; both groups contain tied
recorded day measurements. No jitter or selective deletion was applied.
Melanoma has individual right censoring unsupported by this completed BPCS API.

The selected alternative is the published table of 39 appliance failures in
Shi, Yan and Peng (2025), DOI 10.1016/j.heliyon.2025.e41784, Table 6, attributed
to Nelson (1982). The article XML is saved in `real_data/original`. This is a
verified transcription from a research article; original laboratory records and
the original book were not independently accessed.

There is one cohort and no reported censored devices: all 39 rows have known
failure causes and distinct recorded times. The complete-data special case
therefore has n=m=39 and zero withdrawals. Native cause delta=1 (mechanism 5)
maps to cause 1; native delta=0 (other failure mechanisms) maps to cause 2.
The 26/13 counts are preserved. No artificial second block was created.

Times in thousands of cycles are divided by 5, equivalently cycles divided by
5000. The maximum is 4129 cycles. This fixed analysis scale is not a proven
physical lifetime bound, so the application is conditional on that scale. The
source discusses possible dependence of mechanisms; independent latent causes
and a common shape remain working assumptions, not verified scientific facts.

All six fits passed their numerical checks. All Bayesian primary first and
second moments are finite. The unrestricted MLE gives a=1.024897,
b1=1.256707, b2=0.628354. The restriction pools b1=b2=0.942531 and forces the
cause-1 probability to 0.5, versus 26/39 observed. The survival fit's descriptive
maximum CDF discrepancy is 0.089670; no calibrated KS p-value is claimed.

Full estimates, supported intervals, Bayesian diagnostics and draws, empirical
survival, fitted survival/hazard curves and selected cumulative incidences are
saved separately. No AB, MSE or coverage is calculated for unknown real truth.
This validates execution on the complete-data special case, not the scientific
adequacy of the model or a genuine progressive withdrawal design.

One initial draw-storage call reversed the names/draws arguments. The error was
fixed, the numeric draws were restored from their lossless decimal strings and
verified against stored posterior means. The original artifact and successful
estimator record remain preserved. The authoritative corrected artifact is
listed in `real_data/diagnostics/appliance_draw_artifact_correction.json` and in
`real_data/estimates/appliance_method_comparison.csv`.

## QA and exclusions

Focused generator/storage/analysis/budget tests: 46 passed. Full existing test
suite: 443 passed in 63.96 seconds. The real-data saved draws were reloaded and
their shapes, positivity, restricted support and posterior means verified.
Report.tex, PDFs and LaTeX were neither edited nor compiled.

Changed implementation files: `experiments/simulation_runner.py` (budgeted fit
execution); new `experiments/simulation_budget.py`,
`experiments/submission_supervisor.py`, `experiments/submission_finalize.py`,
`experiments/real_data_validation.py`, `experiments/appliance_validation.py`,
`configs/simulation_submission.json`, and `tests/test_simulation_budget.py`.
This guide and the separate results trees were added. Existing estimator
mathematics, pilot/main configurations and report sources were preserved.

Final resumed supervisor duration: 900.926 seconds. Successful Bayesian fits:
18 independent unrestricted, 19 independent within block, 19 hierarchical
unrestricted, and 13 hierarchical within block. Four restricted hierarchical
fits failed diagnostics; 167 planned Bayesian method fits remain uncompleted.
All four Bayesian methods succeeded together on 6 strict-order, 6 equality and
1 violation dataset. There were 12 interruption records across the stopped and
resumed runs. Final independent recomputation again passed all 702 point-metric
and 756 interval-metric groups. All three supervised workers exited cleanly.
