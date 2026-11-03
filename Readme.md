# Kumaraswamy competing risks under block progressive censoring

Python implementation of classical and Bayesian estimation for two competing
failure causes under block progressive Type-II censoring (BPCS).

Each block has two independent latent Kumaraswamy lifetimes, a shared positive
shape parameter `a`, and block-specific positive parameters `b1` and `b2`.
The order-restricted methods impose **`b1 <= b2` within each block**. There is
no restriction across blocks.

The repository includes the model, validated data structures, likelihood,
unrestricted and restricted MLE, supported uncertainty procedures, four
Bayesian methods, a reproducible simulation pipeline, tests and saved results.
The simulation is partially completed; its exact successful and failed counts
are recorded in the saved manifest.

## Start here

1. Follow **Installation** below.
2. Run the tests and the small teaching examples.
3. Read the saved results without launching another simulation.
4. Use the simulation commands only when you intend to perform new computation.

For a guided explanation of the modules, see [Code guide](docs/CODE_GUIDE.md).
The original assignment is [Project statement](docs/PROJECT_STATEMENT.md).
The documentation index is [docs/README.md](docs/README.md).

## Installation

Use Python **3.12** and run commands from the repository root. The storage and
time-budget mechanisms target **macOS/Linux local filesystems** and use POSIX
features. A native Windows execution environment is not supported by these
mechanisms.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

NumPy and SciPy provide the numerical routines; pandas supplies data tables;
Matplotlib is available for plotting; pytest runs the checks.

For the exact recorded dependency versions, use the lock file instead:

```sh
.venv/bin/python -m pip install -r requirements-lock.txt
```

The lock file describes the recorded environment. It does not replace the
platform and Python-version requirements. No package installation of this
repository is needed: run the modules from this directory with `python -m`.

## Folder structure

```text
Under-Construction/
├── src/                         # Statistical implementation and data contracts
├── experiments/
│   ├── examples/                # Small teaching demonstrations
│   ├── validation/              # Independent numerical verification
│   ├── real_data/               # Source screening and real-data estimation
│   ├── execution/               # Submission supervision and publication
│   ├── legacy/                  # Historical, date-specific execution scripts
│   ├── simulation_*.py          # Stable simulation engine and command line
│   └── *_example.py, ...        # Compatibility entry points for older commands
├── configs/                     # Study designs and prior configurations
├── data/examples/               # Hand-constructed teaching observations
├── results/                     # Saved datasets, estimates, diagnostics, CSVs
├── tests/                       # Numerical and workflow regression tests
├── docs/                        # Derivations, usage guides and historical audits
├── References/                  # Supplied research references
├── requirements.txt             # Dependency requirements
└── requirements-lock.txt        # Recorded exact environment
```

The short compatibility files in `experiments/` forward to the grouped
implementations. Existing imports and commands continue to work. Statistical
and simulation-engine modules retain their original paths because saved
campaigns fingerprint those source files.

## Run the tests

```sh
.venv/bin/python -m pytest -q -W error
```

A smaller workflow check is:

```sh
.venv/bin/python -m pytest -q tests/test_simulation_generator.py tests/test_simulation_pipeline.py tests/test_simulation_analysis.py tests/test_simulation_budget.py tests/test_priority_supervisor.py
```

Tests include distribution identities, risk sets, likelihood derivatives,
independent optimizer comparisons, order restrictions, uncertainty calculations,
posterior conditions, sampler diagnostics, reproducible storage and resumption.
They use small test fixtures; they do not launch the full submission study.

## Teaching examples

These examples use the same small hand-constructed dataset in
`data/examples/two_blocks.json`; they are not real-data or simulation results.

```sh
.venv/bin/python -m experiments.examples.stage2_example
.venv/bin/python -m experiments.examples.stage3_example
.venv/bin/python -m experiments.examples.stage4_example
.venv/bin/python -m experiments.examples.order_restricted_example
.venv/bin/python -m experiments.examples.classical_inference_example
```

They demonstrate validated blocks, likelihood evaluation, profile/direct MLE,
within-block pooling, and classical intervals/reliability, respectively.

The Bayesian example performs MCMC and therefore takes longer:

```sh
.venv/bin/python -m experiments.examples.bayesian_example --output results/local/bayesian_example.json
```

The example writes the selected output path. Use a fresh local path to keep
published outputs separate. Old commands such as
`python -m experiments.stage4_example` remain valid.

## Use the statistical API

```python
from experiments.examples.stage2_example import load_example
from src.mle import fit_mle
from src.order_restricted import fit_order_restricted_mle
from src.uncertainty import parameter_inference

observations = load_example()
unrestricted = fit_mle(observations)
restricted = fit_order_restricted_mle(observations)

print(unrestricted.status, unrestricted.converged)
print(unrestricted.a, unrestricted.b_by_block)
print(restricted.diagnostics["active_constraints"])

if unrestricted.converged and unrestricted.status == "interior_mle":
    intervals = parameter_inference(observations, unrestricted)
    print(intervals.intervals["a"])
```

Always check status and convergence before interpreting estimates. Boundary
suprema, unbounded likelihoods, numerical failures and unsupported intervals
are represented explicitly. The Bayesian API and prior specification are
explained in [Bayesian guide](docs/BAYESIAN_GUIDE.md).

## The six estimation methods

| Method ID | Estimator |
|---|---|
| `mle_unrestricted` | Unrestricted maximum likelihood |
| `mle_within_block` | MLE with within-block ordering |
| `bayes_independent_unrestricted` | Independent-prior Bayesian estimator |
| `bayes_independent_within_block` | Independent-prior estimator with ordering |
| `bayes_hierarchical_unrestricted` | Hierarchical Gamma Bayesian estimator |
| `bayes_hierarchical_within_block` | Hierarchical Gamma estimator with ordering |

Bayesian posterior propriety, first/second moments, R-hat and bulk/tail ESS
are checked. Diagnostic failures are retained and excluded from valid results.
The submission configuration uses four chains, 1,000 warmup iterations and
2,000 retained draws per chain, R-hat <= 1.01 and minimum ESS 400.

## Inspect the saved submission results

Start with these files; reading them does not run estimators:

| Path under `results/simulation_submission/` | Contents |
|---|---|
| `manifest.json` | Current execution audit, exact counts and analysis reference |
| `metadata/latest_analysis.json` | Current versioned export manifest |
| `datasets/` | Saved observed datasets and integrity metadata |
| `estimates/replication_estimates.csv` | Individual estimates and validity flags |
| `estimates/replication_intervals.csv` | Individual confidence/credible intervals |
| `summaries/parameter_performance.csv` | Means, signed/absolute bias, MSE and valid counts |
| `summaries/interval_performance.csv` | Coverage, average width and valid counts |
| `summaries/relative_mse_comparison.csv` | Comparisons on common valid datasets |
| `summaries/method_success_rates.csv` | Successful, failed and missing fit counts |
| `diagnostics/bayesian_diagnostics.csv` | R-hat, ESS, MCSE and moment checks |
| `diagnostics/failure_log.csv` | Recorded failures and unrun tasks |

The submission has 60 saved datasets: 20 each for `strict_order_late`,
`equality_late` and `violation_late`. Both MLE methods have 60 successful fits.
The saved manifest currently reports Bayesian successes of 31, 31, 31 and 27
in the order shown above, with 11 restricted-hierarchical diagnostic failures.
All four Bayesian methods succeeded together on 12 strict-order, 13 equality
and 1 violation dataset. These are **partial, exploratory results**, not a
completed 60-dataset Bayesian comparison. Use the manifest for authoritative
counts if results are extended later.

JSON research records use an integrity envelope with `payload` and `sha256`.
CSV convenience paths and `manifest.json` can be symlinks to immutable versions;
keep their referenced directories when copying the repository.

## Run or resume a simulation

Check the command interface first:

```sh
.venv/bin/python -m experiments.simulation_runner --help
```

Configuration sizes differ substantially:

| Configuration | Scenarios × repetitions | Planned datasets |
|---|---:|---:|
| `configs/simulation_submission.json` | 3 × 20 | 60 |
| `configs/simulation_pilot.json` | 12 × 50 | 600 |
| `configs/simulation_main.json` | 12 × 2,000 | 24,000 |

For a new local study, copy a configuration, give it a **new output directory**,
then run generation, fitting and analysis separately. For example:

```sh
.venv/bin/python -m experiments.simulation_runner generate --config configs/simulation_submission.json --output results/local_study
.venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_submission.json --output results/local_study --methods mle_unrestricted mle_within_block --budget-seconds 900
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.simulation_runner fit --config configs/simulation_submission.json --output results/local_study --methods bayes_independent_unrestricted bayes_independent_within_block bayes_hierarchical_unrestricted bayes_hierarchical_within_block --budget-seconds 900
```

The fitting command prints its campaign ID. Use that exact ID for analysis:

```sh
.venv/bin/python -m experiments.simulation_runner analyze --config configs/simulation_submission.json --output results/local_study --fit-id YOUR_PRINTED_CAMPAIGN_ID
```

`generate` persists observations before estimation. `fit` reloads saved data;
it does not generate replacements. `analyze` reads saved fits without sampling.
`run` combines the stages and can be expensive; it is not needed to view results.

Repeating a fit command with the same campaign settings reuses committed
results, including recorded failures. Interrupted tasks without a final record
can resume. `--attempt` creates a separate campaign for an explicit retry;
it never overwrites prior success. Source/configuration/environment changes
can produce a different campaign ID. Do not delete locks while workers run;
`--recover-locks` is only for confirmed stopped workers.

`--budget-seconds` supervises fits using POSIX alarms and a monotonic clock.
It is not a deadline for reading documentation or publishing summaries.
Historical runs demonstrated that this does not guarantee a strict wall-clock
limit across machine suspension. The dated scripts in `experiments/legacy/`
retain their original behavior and are not general-purpose launchers.

For parallel submission execution, the reusable launcher is
`experiments.execution.submission_supervisor`; inspect its `--help`, supply a
new future `--deadline-unix`, and use `--workers-per-scenario 1` for three
workers on the three-scenario configuration. Limit BLAS/OpenMP threads to one.
See [Simulation guide](docs/SIMULATION_GUIDE.md) for complete schemas, seed
rules, optional draw storage, prior configuration and interval exclusions.

## Real-data application

The saved application uses 39 appliance failures in one complete-data block.
Start with `results/real_data/manifest.json`,
`results/real_data/appliance_source_metadata.json` and
`results/real_data/validation/completed_validation_summary.json`.

Original source files are in `results/real_data/original/`; processed observations,
estimates, intervals, diagnostics and validation are stored separately. Both MLE
methods and all four Bayesian methods completed successfully on this application.
The manifest also identifies the corrected draw artifact from the initial run.

The grouped commands, when reproduction is intended, are:

```sh
.venv/bin/python -m experiments.real_data.real_data_validation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.real_data.appliance_validation --budget-seconds 900
```

They require the saved source files in `results/real_data/original/` and must
be launched from the repository root. They preserve existing committed results.
Hoel/melanoma screening records describe rejected candidates, not the selected
application. Times in the appliance table were divided by 5 (thousands of
cycles), equivalent to dividing cycles by 5,000. This is a chosen analysis scale,
not a verified physical lifetime bound. Independence and common-shape assumptions
remain working assumptions. Complete-data validation is not evidence of a real
progressive-withdrawal experiment. No real-data AB/MSE/coverage is calculated.

## Submission and reproducibility

Include `src/`, `experiments/`, `tests/`, `configs/`, `data/`, `docs/`,
`Readme.md`, both requirements files and the supplied references. Include
`results/` when saved datasets/results are required; preserve symlink targets,
immutable versions, source metadata and failure records together.

Do not include `.venv/`, `.git/`, `__pycache__/`, `.pytest_cache/` or `.DS_Store`.
The receiving machine should create its own environment using the instructions
above. Keep the repository folder structure intact and run from its root.

The report and reference PDFs are separate research artifacts. This code
organization change does not edit them. Historical milestone counts and dated
execution notes are retained in `docs/`; current usage is documented here.
