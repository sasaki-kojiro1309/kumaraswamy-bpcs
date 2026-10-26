# Final classical estimation code audit

Date: 2026-10-26. Scope: the documented common-a, block-specific-b Kumaraswamy
BPCS model with known causes and fixed, noninformative completed schemes.

## Assessment

The implemented statistical formulas are consistent with the likelihood and
model assumptions. **No error in the mathematical derivations or MLE algorithms
was found.** Five reproducible numerical/input-handling defects were corrected.
All 325 tests pass, with zero failures or skips, including warnings-as-errors.
This supports reliability within the documented numerical and statistical
limits; it is not a guarantee for every float64 input or of small-sample coverage.

## Confirmed defects and corrections

| Severity | File / affected operation | Reproduction before correction | Correction and justification |
| --- | --- | --- | --- |
| Medium, extreme parameters | `src/model.py`, log-power near t=1 | Kw(a=1e-320,b=1) at t=nextafter(1,0) returned NaN log density and infinite hazard although both are representable | If a*(-log t) is subnormal, evaluate its logarithm as log(a)+log(-log t). The difference between log(1-exp(-v)) and log(v) is O(v), below log-scale precision here. |
| Medium, extreme parameters | `src/model.py`, scaled tail log survival | Kw(a=500,b=1e308).cdf(.2) returned 0 instead of approximately 3.273390607896e-42; subnormal powers also lost relative precision before scaling | For subnormal exp(-v), compute c*log(1-exp(-v)) through -sign(c)*exp(log(abs(c))-v). Neglected relative corrections are O(exp(-v)); c=0 contributes exactly zero. The same helper serves log density and log survival. |
| Medium | `src/uncertainty.py`, covariance symmetrization | information [[1e-308]] solved to finite variance approximately 1e308, but (V+V.T)/2 returned infinity after the finite check | Average V/2+V.T/2; preserve the diagonal exactly so subnormal valid variances survive. Existing definiteness/conditioning checks and rejection of zero diagonals remain. |
| Medium, reporting | `src/uncertainty.py`, raw normal Wald interval | estimate=1, SE=1e308 produced infinite endpoints with status available | Check raw endpoints for finiteness and return status unavailable with None endpoints on overflow. Log intervals also explicitly reject numerically saturated zero estimates. |
| Low | `src/uncertainty.py`, malformed parameter pair | a zero-dimensional NumPy array used as a b pair raised an unintended TypeError while iterating | Require a one-dimensional container before iterating, yielding the documented ValueError. |

Every defect has a focused regression test. Ten added regression cases cover
these defects and their relevant variants; six additional cases check off-fit
Hessians for three more a values and both parameterizations. Existing assertions
were not weakened. Public APIs, parameter conventions and scientific assumptions
are unchanged. No epsilon, clipping, ridge or pseudoinverse was introduced.

## Mathematical audit

Inspected all source modules, all seven test modules, all six numerical example/
validation modules, and the model, derivation and research documentation.
The Bayesian file remains an unimplemented integration placeholder. Verified:

- D1–D7: normalized density, survival/CDF, hazard, minimum distribution,
  independent competing-risk subdensities, probabilities, incidence and median.
- L1–L8: conditional risk-set argument, C=product Y, withdrawals' survival
  factors, cause-sequence convention, block independence, full log likelihood.
- M1–M8: raw/log scores, conditional b=d/W, profile/envelope scores, strict
  concavity in a, finite maximum conditions, exact zero-count closure limits,
  all-single-failure unbounded likelihood, and numerical failure distinctions.
- R1–R6: p<=1/2 equivalence, count-based pooling, retention of feasible pairs,
  unchanged a and B, count-only nonnegative likelihood penalty, raw and
  proportion-coordinate KKT equations, active equality versus zero boundary.
- I1–I7: observed negative Hessians in both orders; all zero/nonzero entries;
  nonlinear Hessian score correction away from stationary points; Jacobian
  covariance transformation at the unrestricted MLE; Cholesky conditioning;
  normal/log/logit interval formulas; exact binomial confidence-set intersection
  and its marginal coverage argument; reliability gradients and delta variances.

No unrestricted Wald inference is silently assigned to active restricted cause
parameters. Positive order equality remains an attained estimate with separate
active diagnostics; zero b remains a boundary supremum. Restricted individual-b
and incidence intervals remain unsupported. Time-component inference retains
its separate regular time-model assumptions.

## Independent numerical verification

All values below were obtained by executed commands, not inferred from previous
reports. Fixed teaching data and deterministic test inputs were used throughout.

| Check | Actual result |
| --- | --- |
| Profile versus direct unrestricted joint log-likelihood | absolute difference 8.881784197001252e-16 |
| Analytical restricted versus independent scaled SLSQP log-likelihood | absolute difference 1.9206858326015208e-14 |
| Restricted SLSQP versus analytical a | absolute log ratio 1.643007765569751e-8 |
| Restricted SLSQP parameter error / block total | maximum 6.577650406777244e-8 |
| Restricted SLSQP starts | all three solver-successful and independently verified |
| Transformed covariance versus original-information linear solve | max absolute error 5.551115123125783e-17 |
| Additional likelihood-difference Hessian checks, a=.35,1.8,4.5, both coordinates | max absolute error 4.683422389462066e-6 with absolute step 5e-5 |
| Tiny-shape log survival versus 420-digit Decimal evaluation | relative error 1.469649978907389e-16 |
| Subnormal-tail log survival, a=460 and 500, b=1e308, t=.2, versus 420-digit evaluation | maximum relative error 1.0386960861660578e-14 |

Central differences have truncation/cancellation error and explicit test
thresholds. Existing independent checks also include normalized probability
integrals, raw scores and envelope identities, population-quantile recovery,
reliability gradient differences, binomial-outcome coverage enumeration, and
SLSQP checks across extreme time values and both zero-count directions.
The high-precision comparisons used Python's standard-library Decimal with
precision 420 and exact conversion of the float inputs, without adding a
runtime dependency or changing the estimators.

## Reproduction and automation

From the repository root:

```sh
.venv/bin/python -m pytest -q -W error
.venv/bin/python -m experiments.stage2_example
.venv/bin/python -m experiments.stage3_example
.venv/bin/python -m experiments.stage4_example
.venv/bin/python -m experiments.order_restricted_example
.venv/bin/python -m experiments.classical_inference_example
.venv/bin/python -m experiments.within_block_validation --output /tmp/final_bpcs_validation.json
.venv/bin/python -m compileall -q src tests experiments
git diff --check
```

The suite reports **325 passed** (309 at baseline plus 16 added cases), zero
failed, zero skipped. All six examples were also executed with `-W error`,
returned exit code 0 and produced no stderr. Compilation and whitespace checks
passed. The JSON was written outside tracked results to preserve the previous
validation artifact. The printed estimates and statuses agree with the recorded
teaching examples; unavailable/empty intervals retain their explicit status.

`.github/workflows/tests.yml` adds push, pull-request and manual test runs on
Python 3.12 with read-only repository permissions. It installs requirements and
runs pytest with warnings as errors. Hosted execution has not occurred in this
local task; it requires pushing the workflow. Linux dependency resolution and
hosted CI therefore remain externally unverified.

## Documentation, efficiency and preserved scope

Updated Readme.md, DERIVATIONS.md, MODEL_SPEC.md, WITHIN_BLOCK_VALIDATION.md and
RESEARCH_NOTES.md. Older test counts and stage descriptions are explicitly
historical; current supported uncertainty and pending work are identified.
No performance optimization or speedup is claimed: the audit found no need to
alter optimizers or matrix algorithms for performance. Arithmetic corrections
use the existing vectorized/log-scale and Cholesky approaches.

MLE, likelihood, censoring and restricted-estimator algorithms were preserved.
Only `src/model.py` and `src/uncertainty.py` production arithmetic/validation
changed. `tests/test_model.py` and `tests/test_uncertainty.py` gained cases.
The workflow and this report are new. Report.tex, reference PDFs, scientific
model assumptions, data fixtures and unrelated files were not modified.

## Remaining limitations

**Numerical:** finite float64 range, rounded/saturated probabilities and medians,
poorly conditioned information, finite bracket budgets, direct-optimizer guards
and nearly coincident observations can prevent estimation or regular covariance.
An unrepresentable finite MLE remains a numerical failure, separate from a
mathematically unbounded likelihood or zero-parameter supremum. The restricted
production fit still depends on successful unrestricted profiling. Ordinary
probabilities/densities may legitimately underflow/overflow; log APIs should be
used for estimation. Deterministic tests do not cover every possible input.

**Statistical, intentionally unsupported:** restricted individual-b intervals;
restricted/zero-boundary incidence intervals; ordinary full-parameter Wald
inference at a zero cause boundary; simultaneous probability sets or reliability
bands; expected-information calculations. Wald/delta intervals remain marginal
large-sample approximations, especially tentative with seven teaching failures.
Exact binomial intersections can be empty and require the correct order and
fixed-count, known-cause, noninformative-withdrawal model for coverage.

Published-table reproduction and empirical coverage studies remain pending.
No Bayesian estimation, new restriction type, new estimator, simulator,
real-data analysis or paper restructuring was undertaken.
