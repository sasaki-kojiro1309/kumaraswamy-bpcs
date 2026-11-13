# Kumaraswamy competing risks under block progressive censoring

Research code for **Classical and Bayesian Estimation for Block Progressively
Censored Competing Risks Data from the Kumaraswamy Distribution**.

The completed simulation contains 60 original datasets and **360 verified valid
fits**: 60 for each estimator, with all six methods compared on the same 20
original dataset IDs in each of three scenarios. Historical partial runs and
failed attempts are retained as evidence, separately from the final registry.

## Model and estimators

In block i, two independent latent lifetimes follow Kw(a, b_i1) and Kw(a, b_i2).
The shape a is common to all blocks. The observed failure is their minimum,
with known cause and survival (1-t^a)^(b_i1+b_i2). Fixed progressive Type-II
withdrawals operate separately in each block. Restricted methods impose
b_i1 <= b_i2 within each block; there is no cross-block order constraint.

| Method ID | Procedure |
|---|---|
| `mle_unrestricted` | Profile maximum likelihood |
| `mle_within_block` | Order-restricted maximum likelihood |
| `bayes_independent_unrestricted` | Independent scale-prior Bayes |
| `bayes_independent_within_block` | Independent scale-prior Bayes with ordering |
| `bayes_hierarchical_unrestricted` | Hierarchical Gamma-prior Bayes |
| `bayes_hierarchical_within_block` | Hierarchical Gamma-prior Bayes with ordering |

The original priors, seeds and numerical settings are preserved. Bayesian
validity requires the documented propriety/moment checks, four chains,
R-hat <= 1.01 and bulk/tail ESS >= 400 for every monitored quantity.
Unavailable intervals and failed attempts are explicit, never fabricated.
Mathematics and limitations are in [the documentation index](docs/README.md).

## Repository

```text
src/                 Statistical models, likelihoods, estimators and intervals
experiments/         Simulation/completion entry points and workflow packages
  examples/          Small teaching demonstrations
  real_data/         Source screening and appliance-data estimation
  validation/        Independent numerical checks
  execution/         Historical submission supervision and publication
  legacy/            Date-specific execution implementations, kept for provenance
configs/             Simulation designs and prior specifications
data/examples/       Hand-constructed example observations
results/             Original observations, raw fits, diagnostics and final exports
tests/               Automated statistical and workflow regression tests
docs/                Mathematical derivations, workflow guides and dated audits
References/          Supplied research references
output/pdf/          Preserved valid report builds and compilation provenance
Report.tex           Current manuscript source
README.md            Installation and authoritative workflow entry points
```

Stable source modules and result paths are deliberately unchanged: saved
campaigns fingerprint their source bytes and reference immutable records.
Referenced flat compatibility modules remain for existing tests and commands;
new work should use the canonical entry points below.

## Install and test

Use Python 3.12 on macOS/Linux and run commands from the repository root.
The storage and execution-budget code uses POSIX filesystem/process features.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q -W error
```

`requirements.txt` lists direct dependencies; the lock file records the tested
versions. The suite uses small fixtures, not a rerun of the submission study.

A small classical estimation example, using only teaching observations:

```sh
.venv/bin/python -m experiments.examples.stage4_example
```

Other demonstrations live in `experiments/examples/`. See the
[code guide](docs/CODE_GUIDE.md) for their purposes and the Python API.

## Completed simulation results

Start at **[results/simulation_completion_v1/latest](results/simulation_completion_v1/latest/)**.
It points to the authoritative final analysis snapshot. Preserve the symlink
and its target when copying the project.

| File in `latest/` | Contents |
|---|---|
| `SUMMARY.md` | Scenario-wise and parameter-wise results and interpretation |
| `valid_fit_counts.csv`, `selected_fits.csv` | Final counts and selected-record provenance |
| `datasets.csv` | Original dataset IDs, hashes and exact random seeds |
| `six_method_paired.csv`, `paired_dataset_ids.csv` | Six-way comparisons on identical IDs |
| `relative_mse_comparisons.csv` | All 1,755 paired method contrasts |
| `parameter_performance.csv` | Bias/MSE with explicit denominators |
| `interval_performance.csv` | Confidence/credible coverage and width |
| `bayesian_diagnostics.csv`, `mcmc_summary.csv` | Parameter-wise MCMC diagnostics |
| `independent_qc.json`, `validation.json` | Numerical and integrity checks |
| `figures/`, `markdown/` | Final plots and readable table companions |

Raw historical fits and all 60 observations remain under
`results/simulation_submission/`; successful completion fits are in
`results/simulation_completion_v1/attempts/`. Required draw checkpoints remain
under `results/simulation_submission/draws/completion_v1/`. Do not delete the
historical campaign: the final registry reuses its valid fits.

Read [the completion record](docs/SIMULATION_COMPLETION.md) for the full artifact
index, design, findings, diagnostics, runtime interpretation and limitations.
The old submission manifest describes the historical partial campaign, not the
final 360-fit status.

Verify the saved results without sampling or compiling:

```sh
.venv/bin/python -m experiments.completion_qc
```

Regenerate the analysis from saved fits only, without invoking the PDF compiler:

```sh
MPLCONFIGDIR=tmp/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
from experiments.completion_analysis import analyze
from experiments.simulation_config import load_config
analyze(load_config('configs/simulation_submission.json'))
PY
```

This regenerates CSV/Markdown tables, figures and the simulation LaTeX input.
It performs no dataset generation or estimator fitting. The existing
`experiments.completion_study analyze` command additionally compiles the current
manuscript and requires a local Tectonic installation with its TeX resources.
The valid saved build is [output/pdf/Report.pdf](output/pdf/Report.pdf).
The later-edited `Report.tex` is preserved; its broken root PDF was removed at
the user's request. The historical artifact manifest's root-report hashes
therefore describe an earlier manuscript build, not the current source.

## Simulation and real-data workflows

The completed study requires no more fitting. For deliberate future execution,
inspect `.venv/bin/python -m experiments.completion_study --help` or
`.venv/bin/python -m experiments.simulation_runner --help` first. The
[simulation guide](docs/SIMULATION_GUIDE.md) explains generation, fitting,
resumption, immutable storage and explicit retry campaigns. The submission
configuration plans 60 datasets; pilot/main configurations plan 600/24,000.

The selected real-data application uses 39 appliance failures. Its saved
results, source material and transformations are under `results/real_data/`;
start with `manifest.json` and `validation/completed_validation_summary.json`.
When an intentional real-data rerun is wanted:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.real_data.appliance_validation --budget-seconds 900
```

`experiments.real_data.real_data_validation` preserves the separate Hoel and
melanoma compatibility-screening workflow; those are rejected candidates, not
additional fitted applications. The appliance application is conditional on
its documented time scale and independence assumptions. No real-data bias,
MSE or coverage is claimed for unknown truth.

## Reproducibility and submission

Keep source, tests, configurations, example inputs, references, report source,
and the complete results trees together. Hash envelopes, symlink targets,
original seeds, historical failures and draw artifacts are part of the evidence.
Keep `.venv/`, caches, editor files and `tmp/` out of a source distribution.
No package installation of this repository itself is necessary.
