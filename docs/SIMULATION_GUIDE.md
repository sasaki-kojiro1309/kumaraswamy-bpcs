# Kumaraswamy BPCS simulation framework

**Current completion (13 November 2026):** all 360 original combinations have
valid selected fits. See [Simulation completion](SIMULATION_COMPLETION.md) for
the final registry, paired comparisons and the local saved-result reproduction
command. The original submission pointers below remain a preserved historical
campaign. Statements below about an unexecuted framework describe its original
implementation milestone, not the final verified completion.

The framework has been executed on the submission configuration. Saved datasets,
partial Bayesian results, classical fits and versioned summaries are available
in `results/simulation_submission/`. Its `manifest.json` records the authoritative
counts; a planned configuration size does not imply that every fit succeeded.
The [main README](../Readme.md) contains current setup and execution instructions,
and the [code guide](CODE_GUIDE.md) explains the grouped workflow packages.

## Components and data flow

| File | Responsibility |
|---|---|
| experiments/simulation_config.py | Strict explicit configuration validation, withdrawal plans, stable IDs and seed domains |
| experiments/simulation_generator.py | Two independent latent inverse draws per unit, actual sequential withdrawals, observed-only serialization and validated loading |
| experiments/simulation_storage.py | Atomic immutable commits, hash verification, task locks, optional compressed draws and latest-export pointers |
| experiments/simulation_adapters.py | Six existing estimator integrations, moment/propriety/diagnostic statuses, true values, interval collection |
| experiments/simulation_runner.py | Separate generate, fit, analyze and full-pipeline commands; campaign identity and resume orchestration |
| experiments/simulation_analysis.py | Replication CSVs, conditional AB/MSE, interval CP/AW, rates, runtime, pooling and paired comparisons |

Data flow is **configuration -> generated observed dataset -> atomic disk
commit -> disk reload/validation -> the six adapters -> immutable task records
-> offline analysis**. In full-pipeline mode the generation stage completes
before the fitting stage begins. A fit-only command cannot generate a dataset.
An analysis command cannot invoke an estimator or sampler.

The core runner is sequential. Separate workers can handle explicitly disjoint
scenario/method selections, with exclusive task locks preventing duplicate
commits. Run one analysis publisher at a time. The optional
`experiments.execution.submission_supervisor` launches scenario workers; its
worker count and future deadline must be supplied deliberately. Filesystem
operations target POSIX local
filesystems (macOS/Linux) with atomic hard links, symlinks and directory fsync;
unsupported filesystems fail explicitly rather than weakening atomicity.

## Statistical generation

For each unit and cause independently, sample an open-interval uniform U and
calculate

    T = exp(log(-expm1(log1p(-U)/b))/a).

This is [1-(1-U)^(1/b)]^(1/a). Exact uniform endpoints are rejected at the
uniform-draw level; no lifetime is clipped. The unit's minimum and the cause
attaining it are computed. Numerical latent ties, tied minima or values that
round to a support endpoint raise a recorded generation failure. There is no
jitter, epsilon substitution, or automatic regeneration of a failed dataset.

For each fixed BlockScheme, the generator maintains active unit indices:
select the earliest failure among the active units, record its time/cause,
remove it, then use `rng.choice(active, size=r, replace=False)` to withdraw
survivors uniformly. This uses a withdrawal RNG independent of both latent
cause RNGs. It then searches the UPDATED risk set for the next failure.
It does not attach withdrawal counts to the first m sorted complete failures.
The final schedule empties the risk set. Existing Observation, BlockData and
BPCSExperiment constructors validate all observed records.

Latent arrays and withdrawal identities are temporary implementation details.
They are neither serialized nor passed to any estimator. The estimator data
argument is only the existing BPCSExperiment. True parameters are evaluation
metadata, never a warm start, fitted prior or additional observation.

## Explicit scenarios and schedules

Both supplied configurations list 12 scenarios explicitly: four order regimes
crossed with the three requested removal plans, all with a=1.5, two blocks,
n_i=50 and m_i=30. Their cause pairs are:

| Regime | A | B |
|---|---|---|
| strict | (.5,1.5) | (.75,2.25) |
| equality | (1,1) | (1.5,1.5) |
| near_equality | (.95,1.05) | (1.45,1.55) |
| violation | (1.5,.5) | (2.25,.75) |

The totals are B_A=2 and B_B=3 in every regime. This holds the true minimum
lifetime law fixed across regimes; samples need not be identical between regimes.
The six methods within one repetition always use the SAME saved observations.
There is only the within-block b_i1<=b_i2 restriction. In violation scenarios,
restricted-model analyses carry `restricted_model_misspecified=true`, derived
from numeric truth rather than the descriptive regime label. Even if only one
block violates a user configuration, its restricted analysis is flagged.

For w=n-m withdrawals:

- early: (w,0,...,0);
- late: (0,...,0,w);
- distributed: r_j=floor(j*w/m)-floor((j-1)*w/m), j=1,...,m.

The distributed cumulative-floor rule spreads remainders across failure
positions with cumulative discrepancy below one. With w=20,m=30, the sequence
is (0,1,1) repeated ten times. Every plan is validated by BlockScheme. A custom
block may instead supply an explicit `withdrawals` array; supply exactly one
of `plan` or `withdrawals` in input JSON.

Arbitrary positive shapes, a positive number of blocks, distinct safe block
IDs, and valid block-specific sizes/schedules are configurable. To explore
n=30,50,100, different observed fractions or a=.8,1.5,3, explicitly add/change
scenario entries and use a NEW output directory. There is no automatic full
factorial expansion. Invalid designs are rejected before execution.

## Configured study sizes and priors

- configs/simulation_pilot.json: 50 repetitions for each of 12 scenarios;
  600 planned datasets and 3,600 planned method tasks.
- configs/simulation_main.json: 2,000 repetitions for each of 12 scenarios;
  24,000 planned datasets and 144,000 planned method tasks.

These are budgets, not executed counts or permission to run them now.
Both configurations enable all six methods and secondary mission times
(.25,.50,.75). Set `mission_times: []` to omit secondary reliability performance
rows. Auxiliary B_i and p_i rows remain available alongside the primary
(a,b1[A],b2[A],b1[B],b2[B]) rows.

Six method IDs and interfaces:

1. mle_unrestricted -> src.mle.fit_mle(data,...).
2. mle_within_block -> src.order_restricted.fit_order_restricted_mle(data,...).
3. bayes_independent_unrestricted -> src.bayesian.fit(data,prior='independent',restricted=False,...).
4. bayes_independent_within_block -> same prior, restricted=True.
5. bayes_hierarchical_unrestricted -> fit with an explicit HierarchicalPrior, restricted=False.
6. bayes_hierarchical_within_block -> same hierarchy, restricted=True.

All Gamma priors are **shape-rate**. The fixed illustrative default is
 a~Gamma(2,2), kappa_1,kappa_2~Gamma(3,2), lambda_1,lambda_2~Gamma(2,2).
This follows the existing guide's illustrative configuration; it is not claimed
to be elicited or optimal. Two separate sensitivity specifications are supplied:
configs/priors/hierarchical_diffuse.json and hierarchical_concentrated.json.
The names describe illustrative alternatives, not a rigorous ordering of every
induced marginal prior. They replace ONLY the hierarchical hyperprior.

Default MCMC controls retain the established API: 4 chains, 1,000 warm-up and
2,000 retained iterations PER CHAIN, slice width 1, step budget 1,000,
max Rhat=1.01, minimum bulk/tail ESS=400, posterior interval probability .95.
No new MCMC algorithm or stopping rule is introduced. Bayesian diagnostics and
moment limitations are exactly those documented in BAYESIAN_GUIDE.md.

## Identities, random streams and reproducibility

Dataset entropy is SHA-256 over a fixed data-domain tag, master_seed, the
entire normalized scenario definition and the repetition number. SeedSequence
spawns one stream per block, then independent cause-1, cause-2 and withdrawal
streams within that block. The generator uses NumPy's default PCG64 generator.
Saved metadata records the exact integer entropy, domain specification, NumPy
version, source hashes, scenario and block order.

MCMC entropy uses a DIFFERENT domain tag and `mcmc_master_seed`, dataset ID
and method ID. Changing MCMC controls, priors, enabled methods, runtime output
location or repetition budget does not change dataset entropy/identity. Python's
randomized hash() is never used. Exact replay requires the recorded software
versions, code and config; timing naturally varies. Different Bayesian methods
have separate sampler seeds, rather than silently coupling chains.

A fit campaign ID hashes sampling/prior/classical controls, storage policy,
mission times, software/source provenance and an explicit attempt number.
Changing those settings creates a NEW campaign, reusing the saved datasets.
Method/scenario CLI filters do not alter dataset or campaign definitions.
Moving datasets to a compatible machine is allowed for fit-only; the new
fitting environment is recorded in a new campaign identity if different.
Generating additional data in an existing study requires unchanged generator
code and NumPy version. Otherwise use a new output root.

## Dataset schema and permanent storage

Every committed JSON file is an envelope:

    {"payload": {...}, "sha256": "SHA256(canonical payload JSON)"}

Canonical JSON uses full round-trip float serialization, sorted keys and no
NaN/Infinity. Failure times are never rounded. Each dataset payload contains
schema/generator versions, unique dataset ID, scenario ID, repetition, seed
information, true parameters, number of blocks, all initial sizes/failure
counts/schedules, observed ordered times and cause labels, and generator
implementation hashes. There are no latent observations in this schema.

Dataset paths are `datasets/<scenario>/rep_000001.json`. Use
`load_dataset(path)` to obtain `(BPCSExperiment, payload, hash)`; the data
object can be passed directly to any established estimator. With scenario,
repetition and master_seed supplied, the loader checks identity, seed, truth
and scheme compatibility as well as hash and data validity.

A temporary file is fsynced and atomically hard-linked into its final name
without replacement. It is read-only at commit. Existing files must match
exactly; corruption or conflicting content stops processing. Dataset generation
finishes with a read-back validation. Commit/hash checks provide integrity,
not protection against a deliberate attacker who rewrites both data and hash.

There is deliberately no overwrite/regenerate-in-place switch. For a deliberate
new sample use a changed seed and NEW output root. Estimator failure never
triggers regeneration. Generation errors have separate committed records and
are reported separately from the number of successfully saved datasets.

## Resume, partial failures and result layout

The first run creates immutable `config_snapshot.json` and `run_manifest.json`.
Subsequent requests are recorded under metadata/requests. The generation
manifest compares the full generation definitions; incompatible scenarios/seeds
cannot be mixed into an existing root. Repetition counts may be increased while
preserving earlier repetition identities. Each fit campaign also saves its
complete settings and provenance under metadata/fit_configs/<fit_id>.json.

Per-method task records are committed to
`estimates/tasks/<fit_id>/<scenario>/rep_000001_<method>.json` and are checked
against the saved dataset ID and hash on reuse. They retain points, intervals,
statuses, timing, diagnostics, moments, tracebacks and optional draw references.
Completed failures are records too and are not silently retried on resume.
To explicitly retry, change the controls or use `--attempt 2`; old task files
are preserved and the data are reused. An interrupted, uncommitted task can be
rerun from its recorded seed after clearing its stale lock.

Task locks record host, PID and ownership token. `--recover-locks` is an explicit
operator action, ONLY after all previous workers have stopped. Known-live local
owners are refused; termination of remote owners must be established by the
operator. Recovery deletes locks, not committed datasets/results. A crash before
commit can leave an ignored `.pending-*` file; final names cannot contain a
partially written dataset. Never delete completed task records to hide failures.

No posterior draws are saved by default. `save_draws.repetitions` can list selected
repetition numbers, and diagnostic/numerical-failure flags can request available
failed-case draws. Saved parameter draws use deterministic compressed NPZ, with
names and dimensions and a SHA-256 reference in the task record; no pickle.
The existing Bayesian module may return no draws after a numerical failure;
those cannot be recovered by the framework. Priors, sampler controls, seeds,
code/version provenance and data hashes are retained regardless of draw saving.

## Offline aggregation and denominators

Each analysis has an immutable ID based on the selected campaign, source/data/
result hashes, methods, scenarios and repetition budget. Versioned CSVs live in
`<category>/exports/<fit_id>/<analysis_id>/`. Top-level requested CSV paths are
convenience SYMLINKS to the latest published version, not overwrites of previous
results. The transactional registry is metadata/analyses/<analysis_id>.json;
metadata/latest_analysis.json points to it. If publication is interrupted, the
registry and versioned exports remain authoritative. Read one manifest for a
consistent snapshot rather than opening files while another process publishes.

Outputs:

- estimates/replication_estimates.csv and replication_intervals.csv;
- diagnostics/bayesian_diagnostics.csv, failure_log.csv and runtime.csv;
- summaries/parameter_performance.csv, interval_performance.csv,
  method_success_rates.csv, relative_mse_comparison.csv, mcmc_summary.csv,
  active_pooling.csv and generation_counts.csv.

All rows identify campaign/analysis, scenario/method and original dataset where
applicable. Summaries include design/truth metadata and valid counts. Different
fit IDs are never automatically pooled. If multiple campaigns exist, analyze
requires `--fit-id`. Analysis reads the campaign's recorded controls, not its
current environment's MCMC defaults. Missing tasks are explicitly `not_run`;
they remain in the generated-dataset denominator.

For each scenario/method/parameter separately, let V be the valid-point subset:

    signed_bias = mean_V(estimate - truth)
    AB = abs(signed_bias)
    MSE = mean_V((estimate - truth)^2)
    mean_estimate = mean_V(estimate)

The reported SD uses ddof=1 and is unavailable for fewer than two valid points.
`R_generated` counts SAVED datasets, `R_point` counts valid point estimates;
point availability is R_point/R_generated. These are conditional-on-valid
statistics, never labeled unconditional performance. Generation counts/failures
are reported separately, so poor generation cannot disappear from accounting.

A converged, positive-domain classical MLE is a successful attained fit. Boundary
suprema are separately flagged; the default `include_boundary_suprema=false`
excludes their closure point values from AB/MSE. Set it true explicitly to
summarize those values, still with boundary status and no claim of attainment.
Time/exact intervals can be available even when a point estimate is excluded.
Bayesian `fit_success` means its diagnostics passed; valid posterior MEANS also
require the returned first moment to be finite and the estimate to be present.
All-primary-valid counts are separate from diagnostic pass rates. Infinite,
unverified and undefined means are never replaced by a finite chain average.

For valid intervals, CP is mean(covered), and AW is mean(width). Every interval
row records parameter, actual method/family, confidence-versus-credible type,
nominal level, endpoints, coverage indicator, width, availability/validity,
source status, and posterior empirical mass when returned. Counts `R_CP` and
`R_AW` are explicit. Normally equal, they differ if a finite-endpoint set has
an unrepresentable width. Unsupported/unavailable intervals are excluded and
counted; they are not zero-width intervals. A legitimate EMPTY exact set has
coverage zero and set length zero, remains in both denominators, and is never
replaced by equality. MCMC-failed intervals retain endpoints as exploratory
but do not enter CP/AW. Proper posterior intervals may enter those summaries
even if their posterior mean is infinite/unverified. Empirical coverage of a
credible interval is a repeated-sampling evaluation, not a relabeling of the
interval itself as a frequentist confidence interval.

Restricted individual-b Wald intervals and restricted/zero-boundary CIF intervals
remain unsupported. Classical normal/log/logit intervals and exact probability
sets are collected only through the existing uncertainty API. Posterior
intervals are the actual returned shortest contiguous sample intervals, with
its qualified unimodal-HPD-approximation label; they are never clipped
unrestricted intervals. A Bayesian B_i interval is not manufactured because
the current result object does not supply one. Its mean can be obtained as the
sum of two valid finite b means. Secondary reliability means use the existing
draw-wise Bayesian summaries, never nonlinear plug-ins at parameter means.

Failure outputs distinguish unbounded likelihood, boundary supremum, improper
posterior, unsupported prior/data combination, withheld means, numerical failure,
optimizer failure, MCMC diagnostic failure, unavailable intervals and unexpected
software errors. Inference errors can coexist with valid point estimates.
Numerical nonfinite diagnostics serialize to null with their fit failure/status
retained. Rates and timing are separate from parameter-wise validity.

Restricted/unrestricted comparisons use the SAME common-valid dataset IDs and
matching hashes/truth for each method pair and parameter. Their reported ratio is
100*(MSE_U_paired-MSE_R_paired)/MSE_U_paired. Zero MSE_U makes the ratio undefined.
The mean paired squared-error difference is U error squared minus R error
squared. Its Monte Carlo SE is sample SD of those paired differences divided
by sqrt(R_paired), available only for R_paired>=2, and presumes independent
replications and appropriate finite-variance conditions. Both full success counts
and equality/difference of success sets are reported. No universally best method
is selected. Interpret regime-specific tables alongside misspecification,
interval availability, failures, runtime and MCMC diagnostics.

## Exact commands for later execution by the user

Run from the repository root with the existing environment. These commands
have NOT been executed in this implementation task.

### 1. Create pilot datasets only

```sh
.venv/bin/python -m experiments.simulation_runner generate --config configs/simulation_pilot.json
```

### 2. Fit the SAME saved pilot datasets with all six methods

```sh
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_pilot.json
```

The fit command prints its campaign ID. No datasets are generated in this mode.

### 3. Analyze pilot results only

```sh
.venv/bin/python -m experiments.simulation_runner analyze --config configs/simulation_pilot.json
```

With multiple campaigns, append `--fit-id THE_PRINTED_CAMPAIGN_ID`.

### 4. Full main pipeline (explicitly expensive; user-run only)

```sh
.venv/bin/python -m experiments.simulation_runner run --config configs/simulation_main.json
```

### 5. Resume an interrupted run

Rerun the same generate/fit/run command: committed compatible datasets/tasks
are verified and reused. If a crashed worker left locks, AFTER stopping workers:

```sh
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_pilot.json --recover-locks
```

To retry recorded failures in a new campaign rather than overwrite them:

```sh
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_pilot.json --attempt 2
```

### 6. Run the authored tests

```sh
.venv/bin/python -m pytest -q tests/test_simulation_generator.py tests/test_simulation_pipeline.py tests/test_simulation_analysis.py
```

The new adapter tests mock estimator calls; they do not run MCMC. Generator tests
intentionally generate tiny fixtures WHEN YOU EXECUTE THEM. The small uniform
removal check is a sampler-component check, not a performance experiment.
For all existing and new tests (existing Bayesian tests do run sampling):

```sh
.venv/bin/python -m pytest -q -W error
```

### Explicit subset, settings and prior sensitivity examples

```sh
.venv/bin/python -m experiments.simulation_runner generate --config configs/simulation_pilot.json --scenarios strict_order_late equality_late --repetitions 10 --output results/simulation/subset
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_pilot.json --scenarios strict_order_late equality_late --repetitions 10 --output results/simulation/subset --chains 4 --warmup 1500 --draws 3000
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_pilot.json --prior-json configs/priors/hierarchical_diffuse.json --methods bayes_hierarchical_unrestricted bayes_hierarchical_within_block
```

`--mcmc-json controls.json` merges sampling-control overrides, including Rhat/ESS
thresholds; `--mcmc-master-seed` changes sampler entropy only. A `seed` field
inside the MCMC controls is rejected to avoid ambiguous stream control.
Scenario selection and repetition count for fit-only must match datasets that
exist, or have explicit generation-failure records. No missing observations
are generated to satisfy a fit request.

## Delivery status and limits

Only static inspection/editing has been performed. Neither the new modules nor
their tests have been executed or certified. There are no empirical results
from this framework. Default scenarios/prior choices require researcher review;
Bayesian moment conditions can be sufficient-only, and zero-count independent
priors retain the existing conservative unsupported policy. Float64 generation,
fit/sampler range, diagnostics and information conditioning can fail. These
failures are recorded without deleting the dataset or substituting a different
prior. Examine tests and a pilot on your own before a main study. No change to
Report.tex or any classical/Bayesian estimation mathematics is included.
