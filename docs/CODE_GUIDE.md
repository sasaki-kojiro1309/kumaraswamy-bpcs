# Code guide

## Reading order

1. `src/model.py`: Kumaraswamy distribution and independent competing risks.
2. `src/censoring.py`: schemes, observations, completed blocks and validation.
3. `src/likelihood.py`: block and joint likelihoods, including censoring constants.
4. `src/mle.py`: unrestricted profile MLE and independent direct optimization.
5. `src/order_restricted.py`: within-block pooling and constrained estimation.
6. `src/uncertainty.py`: information, covariance, supported intervals/reliability.
7. `src/bayesian.py`: priors, posterior checks, fitting and result interpretation.
8. `src/bayesian_sampling.py`: sampling primitives and convergence diagnostics.

These modules contain the statistical implementation. Their contents are
unchanged by the folder reorganization.

## Workflow packages

| Package | Purpose | Starting module |
|---|---|---|
| `experiments.examples` | Learn each stage on small teaching data | `stage2_example` |
| `experiments.validation` | Independent numerical verification | `within_block_validation` |
| `experiments.real_data` | Screen source compatibility and fit real data | `appliance_validation` |
| `experiments.execution` | Reusable submission supervision/publication | `submission_supervisor` |
| `experiments.legacy` | Preserve historical date-specific runs | Read its README before use |

Use `python -m` with the full module name. For example:

```sh
.venv/bin/python -m experiments.examples.stage4_example
.venv/bin/python -m experiments.execution.submission_supervisor --help
```

The old flat module names are compatibility entry points. On import they expose
the same implementation module, including functions used by existing tests.
On command-line execution they call that module's existing `main()` function.
There is one implementation of each relocated workflow, not a second estimator.

## Simulation engine

The simulation modules remain at their established paths:

| Module | Responsibility |
|---|---|
| `simulation_config.py` | Validate designs, configure priors, derive IDs/seeds |
| `simulation_generator.py` | Generate latent lifetimes, apply withdrawals, load observed data |
| `simulation_storage.py` | Atomic immutable records, integrity hashes, locks, optional draws |
| `simulation_adapters.py` | Translate existing estimator results into common records |
| `simulation_budget.py` | Supervise fitting with elapsed-time alarms |
| `simulation_runner.py` | Generate, fit, analyze or run stages |
| `simulation_analysis.py` | Produce estimates, interval, diagnostic and performance CSVs |

Their source bytes are part of saved campaign/generation identities. Moving or
formatting them would change reproducibility fingerprints, so this presentation
cleanup leaves those files intact.

The pipeline is:

```text
configuration
    -> generated observations
    -> immutable saved dataset
    -> validated reload
    -> existing estimators
    -> saved estimates / intervals / diagnostics
    -> offline analysis and versioned CSV exports
```

Truth belongs to generation/evaluation metadata. Estimators receive only the
validated observed-data object. All methods for a repetition use the same saved
dataset. Missing fits and diagnostic failures remain visible in summaries.

## Completed-study entry points

| Module | Purpose |
|---|---|
| `experiments.completion_qc` | Verify the saved 360-fit study without fitting or compiling |
| `experiments.completion_analysis` | Rebuild matched comparisons, tables and figures from saved fits |
| `experiments.completion_study` | Inventory, resumable completion fitting, or analysis plus PDF compilation |
| `experiments.completion_publication` | Generate manuscript tables and compile the report |

The final registry is `results/simulation_completion_v1/latest/`. It combines
preserved valid historical records with separately identified completion records;
`results/simulation_submission/summaries/` remains a historical partial export.
Current commands are maintained in the root [README](../README.md).

## Data and outputs

- `data/examples/two_blocks.json` is hand-constructed teaching data.
- `results/simulation_submission/datasets/` contains observed simulated datasets.
- `results/simulation_submission/estimates/tasks/` contains immutable fit records.
- `results/simulation_submission/metadata/` tracks campaigns, analyses and audits.
- `results/simulation_submission/summaries/` preserves historical performance CSVs.
- `results/simulation_completion_v1/latest/` contains the final matched tables and plots.
- `results/simulation_completion_v1/attempts/` contains successful and failed completion attempts.
- `results/real_data/original/` preserves downloaded source material.
- `results/real_data/processed/` records the reproducible transformation.

JSON result records use a `payload`/`sha256` envelope. Use
`experiments.simulation_storage.read_json` to verify that envelope when loading
it in Python. The teaching input JSON is a different, plain input schema.

## Tests

The `test_model`, `test_censoring`, `test_likelihood`, `test_mle`,
`test_order_restricted`, `test_uncertainty` and `test_bayesian` modules follow the
statistical layers. `test_within_block_validation` checks the independent
optimizer. `test_simulation_*` covers the workflow and saved-result metrics;
`test_priority_supervisor` checks the historical scheduling policy.

No simulation, real-data fitting or report compilation is required to understand
or review this reorganization. The full test command is in the main README.
