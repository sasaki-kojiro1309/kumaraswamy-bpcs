# Kumaraswamy competing risks under block progressive censoring

Research project in progress. **Completed scope: Stage 1 distribution layer,
Stage 2 BPCS data structures/validation, Stage 3 likelihood construction, and
Stage 4 profile/direct maximum likelihood estimation, plus provisional
within-block order-restricted MLE.** Uncertainty methods,
simulation, and real-data pipelines are not implemented.
Bayesian estimation remains a teammate integration placeholder.

Validation on 2026-10-26: **254 tests passed** (170 Stage 1-4 tests,
52 existing order-restricted tests, and 32 additional audit/stability tests)
using Python 3.12; exact package versions are recorded in
`requirements-lock.txt`.

The original project statement is preserved verbatim in
[PROJECT_STATEMENT.md](docs/PROJECT_STATEMENT.md), and the
three source PDFs remain in `References/`. This repository itself is the project
root; no nested duplicate project folder is needed.

## Model and learning notes

The user-approved provisional model has configurable k blocks, one common a>0,
and block-specific b_i1,b_i2>0 for two independent latent Kumaraswamy causes.
Causes are initially fully observed, baseline parameters are unrestricted, and
the teaching example uses k=2 with different sizes and withdrawal schedules.
Block counts, sizes, and schedules are configurable. See [MODEL_SPEC.md](docs/MODEL_SPEC.md) for the precise
assumptions and their provenance.

Start with [DERIVATIONS.md](docs/DERIVATIONS.md): survival of the observed minimum
is the product of the two latent survivals, so its distribution is Kw(a,b_i1+b_i2).
A known cause-c failure has joint density f_ic(t)S_i,other(t), which integrates
to b_ic/(b_i1+b_i2). This separates latent probabilities from observed competing
failure probabilities. [RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) tracks source
locations, apparent paper notation errors, and what has not been reproduced.

[STAGE1_GUIDE.md](docs/STAGE1_GUIDE.md) explains every distribution function and
the original 20 passing tests. [BPCS_GUIDE.md](docs/BPCS_GUIDE.md) explains the
block experiment, risk-set accounting, data contracts, and two-block example.
The Stage 3 section of DERIVATIONS.md derives the ordered-data censoring
constant, block and combined likelihoods, and full log-likelihood from first
principles, then works through the same two-block data.
The Stage 4 section derives scores, conditional estimates, the profile,
concavity/tail results, boundary behavior, and independent direct optimization.

## Reproduce the current milestones

Use Python 3.10 or newer from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.stage2_example
.venv/bin/python -m experiments.stage3_example
.venv/bin/python -m experiments.stage4_example
.venv/bin/python -m experiments.order_restricted_example
.venv/bin/python -m experiments.within_block_validation --output results/within_block_validation.json
```

`requirements.txt` lists the intended stack. `requirements-lock.txt` records the
exact environment used for the current validation; install that file instead
when reproducing this environment. No random seed is needed for the deterministic
tests and the hand-constructed teaching example. SciPy supplies numerical
quadrature and an independent transformed Beta distribution oracle; NumPy supplies
the distribution implementation. pandas exports observation and summary tables.
Matplotlib is reserved for later analysis stages.

```python
from src.model import CompetingRisks

# Illustrative single-block parameter values, not fitted or published estimates.
block = CompetingRisks(a=2.0, b1=1.0, b2=2.0)
print(block.minimum.sf(0.5))       # 0.421875: block survival
print(block.minimum.hazard(0.5))   # 4.0: block hazard
print(block.minimum.median())     # block median time to failure
print(block.cause_probability(1)) # 1/3: eventual cause-1 failure probability
print(block.cause_cdf(0.5, 1))     # P(X <= 0.5, cause = 1)
```

All times are on a fixed (0,1) scale. Hazard accepts interior times only. Density
is assigned zero at the endpoints; limiting densities can differ. Parameters
are positive finite scalars; time arrays are supported. See the numerical
conventions and float64 limits in DERIVATIONS.md.

## Stage 2: validated block data

`src/censoring.py` separates the fixed `BlockScheme` from `Observation` rows,
validates them together as `BlockData`, and combines any positive number of
uniquely named blocks in `BPCSExperiment`. Unknown causes remain reserved and
are rejected by the known-cause baseline. Both the objects and their internal
sequences are immutable; `.to_frame()` and `.summary_frame()` return fresh tables.

The example in `data/examples/two_blocks.json` has A: n=6,m=3,R=(1,0,2) and
B: n=7,m=4,R=(0,1,0,2). Before-failure risk sets are (6,4,3) and (7,6,4,3).
Both blocks empty at their terminal failures: 13 original units = 7 observed
failures + 6 withdrawals. All times are in (0,1), and each block is independently
ordered. These are teaching data, not simulated, real, or published observations.
Chosen parameters printed by the example illustrate Stage 1 reliability only;
they have not been estimated from these data.

## Stage 3: likelihood evaluation

`src/likelihood.py` reuses the unchanged model and censoring implementations.
It supplies block/joint likelihood and log-likelihood functions, plus the log
censoring constant. With the existing teaching data:

```python
from experiments.stage2_example import load_example
from src.likelihood import loglikelihood

data = load_example()
b = {"A": (1.0, 2.0), "B": (0.8, 1.2)}
print(loglikelihood(data, a=2.0, b_by_block=b))
# -2.833127754473: full log density at these chosen parameters
print(loglikelihood(data, a=2.0, b_by_block=b, include_constant=False))
# -13.332370141561: omits only log C, with C=36288
```

Full constants are included by default for the recorded ordered times and cause
sequence, conditional on fixed schemes. `include_constant=False` omits only the
parameter-independent product of risk sets, retaining all time factors. Prefer
log functions for later inference; ordinary likelihoods can underflow/overflow.
Invalid parameters/mappings raise ValueError, while unrepresentable log
arithmetic raises FloatingPointError. Exact data/model assumptions and API
contracts are in DERIVATIONS.md, (L1)-(L8). No prior or fitting is performed.

## Stage 4: profile and independent direct MLE

`src/mle.py` derives and fits one shared a with separate cause parameters for
each block. Given a, the conditional estimates are b_ic=d_ic/W_i(a), where
W_i(a)=-sum_j(r_ij+1)log(1-x_ij**a). Substitution leaves a one-dimensional
profile. The default method solves its unique score root in log(a) with an
adaptive sign bracket. A separate multistart joint optimizer uses log(a) and
log(b_ic), without starting from the profile fit.

```python
from experiments.stage2_example import load_example
from src.mle import fit_mle, fit_direct

data = load_example()
profile = fit_mle(data)
direct = fit_direct(data)
print(profile.status, profile.converged)
# interior_mle True
print(profile.a, profile.b_by_block, profile.loglikelihood)
# a=1.173959840838; A=(.517708109960,.258854054980)
# B=(.178753429726,.536260289178); log L=1.070678947337
print(direct.loglikelihood - profile.loglikelihood)
# about 9e-16 in the validated environment
```

Check both `converged` and `status`. A zero observed cause count produces an
exact zero closure estimate with `boundary_supremum`: no positive-parameter
MLE is claimed. The unchanged Stage 1/3 classes still require positive b; the
MLE module evaluates the boundary limit by an exact linear likelihood identity.
If every block has just one observed failure, the likelihood is unbounded;
the result has `unbounded_likelihood`, a=None, and no finite parameter vector.
Solver budget/gradient failures and numerical range failures have separate
failure statuses. Diagnostics include active log-parameter scores, boundary
one-sided scores, root or multistart details, and a profile/joint likelihood check.

The teaching example fits the existing hand-constructed data, not a simulation.
Parameter-recovery tests use deterministic distribution-quantile fixtures,
including the complete-data special case of BPCS. No random lifetime/censoring
generator, standard errors, intervals, or Bayesian estimator was added.

## Provisional order-restricted MLE

`src/order_restricted.py` implements only b_i1<=b_i2 independently per block,
using the same likelihood as the unrestricted fit. Given a, put
B_i=m_i/W_i(a), p_i=min(d_i1/m_i,1/2), then b_i1=B_i*p_i and
b_i2=B_i*(1-p_i). The common a and all B_i stay unchanged; violating pairs
are pooled equally. The derivation, zero-count limits, and source comparisons
are in DERIVATIONS.md (R1)-(R6). The order requires future scientific justification.

```python
from src.order_restricted import fit_order_restricted_mle

restricted = fit_order_restricted_mle(data)  # existing validated BPCSExperiment
print(restricted.status, restricted.converged)
print(restricted.b_by_block)
print(restricted.diagnostics["active_constraints"])
```

On the teaching data, a=1.173959840838, A=(.388281082470,.388281082470),
and B=(.178753429726,.536260289178). Only A's equality is active. Full log L
is .900779910542, compared with 1.070678947337 unrestricted. Both fits converge
with positive parameters. Equality is an attained order constraint, not a
zero-parameter boundary. No cause-1 failures in a block instead gives a finite
zero-b1 boundary supremum when a is identified; no cause-2 failures gives an
attained positive equality. All single-failure blocks still give unbounded
likelihood. Check status and convergence before using any result.

The original 52 restricted tests include independent joint constrained SLSQP
verification, invalid inputs, numerical failures, and boundary cases. The audit
adds 32 checks covering exact equality retention, KKT stationarity, survival
invariance, extreme valid times, and a standalone independently scaled SLSQP fit.
The latter uses three fixed starts, no profile-based initial values or pooling
updates, and verifies convergence, feasibility, and projected scores. Its
machine-readable result is `results/within_block_validation.json`.
Run the expanded teaching example with
`.venv/bin/python -m experiments.order_restricted_example`; it prints inputs,
unrestricted/restricted/numerical fits, pooling decisions, cause probabilities,
and unchanged minimum survival. See [the audit report](docs/WITHIN_BLOCK_VALIDATION.md)
for modified files, mathematical checks, numerical comparisons, and limitations.
No later research stage is implemented.

## Roadmap

1. **Stage 1 (complete):** validate distribution primitives and explain
   independent competing-risk identities.
2. **Stage 2 (complete):** block data structure, withdrawal schedule and risk-set checks.
3. **Stage 3 (complete):** derive and validate block/joint likelihoods and constants.
4. **Stage 4 (complete):** scores, conditional estimates, profile and direct MLE,
   convergence diagnostics, boundary/unbounded cases, and deterministic validation.
   **Order-restricted MLE extension complete:** provisional within-block b_i1<=b_i2,
   derivation, exact pooling, boundary diagnostics, and independent numerical checks.
5. **Stage 5 (not started):** uncertainty and compatible published numerical validation.
6. **Stages 6-7:** simulate the actual experimental process and evaluate estimators
   on identical datasets with reproducible configurations, tables, and plots.
7. **Stages 8-10:** assess real-data compatibility, integrate teammate estimators,
   investigate labeled extensions, and produce reproducible results.

Current files: `src/model.py`, `src/censoring.py`, `src/likelihood.py`, `src/mle.py`,
their test modules, `src/order_restricted.py`, `docs/`, the two-block JSON data,
Stage 2/3/4 and order-restricted example scripts, and the teammate placeholder
`src/bayesian.py`. Reserved directories
`data/real/` and `results/` contain no research results. Later modules will be
created when their mathematics and contracts are
established; empty implementation stubs would not constitute completed stages.

## References

- **Reference 1 — Wang et al. (2023):** *Inference for block progressive censored
  competing risks data from an inverted exponentiated exponential model.*
  [Local PDF](References/Reference_1.pdf). DOI: 10.1002/qre.3382.
- **Reference 2 — Wang (2018):** *Inference of progressively censored competing
  risks data from Kumaraswamy distributions.*
  [Local PDF](References/Reference_2.pdf). DOI: 10.1016/j.cam.2018.05.013.
- **Reference 3 — Mahto et al. (2022):** *Inference for partially observed competing
  risks model for Kumaraswamy distribution under generalized progressive hybrid
  censoring.* [Local PDF](References/Reference_3.pdf).
  DOI: 10.1080/02664763.2021.1889999.

The references have different distributions or censoring designs. Neither IEED
formulas nor hybrid-stopping likelihoods are substituted into this model.
