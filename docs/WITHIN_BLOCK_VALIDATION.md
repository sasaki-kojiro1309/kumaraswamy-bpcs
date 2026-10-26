# Within-block order-restricted MLE: completion and audit report

Date: 2026-10-26. Scope: unrestricted MLE and b_i1<=b_i2 independently in
each block, with a common positive a, known causes, independent latent risks,
and fixed completed progressive Type-II schemes within blocks.

**Outcome: complete for this scope and the documented float64 limits.**
The audit began with an existing implementation and 222 passing tests. It
preserves the correct baseline and adds 32 focused checks, for **254 passing
tests**. Across-block or combined orders, Bayesian estimation, confidence
intervals, and simulation studies have not been implemented by this task.

## Files inspected and preserved

Reviewed the distribution, censoring, full likelihood, unrestricted estimator,
restricted estimator, existing derivations/model specification, tests, and
teaching examples. The source files inspected were `src/model.py`,
`src/censoring.py`, `src/likelihood.py`, `src/mle.py`, and
`src/order_restricted.py`. Existing validation included `tests/test_model.py`,
`tests/test_censoring.py`, `tests/test_likelihood.py`, `tests/test_mle.py`, and
`tests/test_order_restricted.py`. Read the relevant likelihood/MLE sections of
the local PDFs and checked the derivation against the actual block model.

SHA-256 comparisons against the start of this audit confirm the first four
source files and all five existing test files are unchanged. In particular,
the unrestricted implementation and its uniqueness proof were preserved.
The proof's series-differentiation assumptions are now explained more explicitly.

## Files modified or added

| File | Change |
| --- | --- |
| `src/order_restricted.py` | Retain already-feasible equal-count pairs exactly; add and enforce relative block-total preservation diagnostics |
| `experiments/order_restricted_example.py` | Print input data, three fitted results, pooling decisions, cause probabilities, convergence details, and unchanged minimum survival |
| `experiments/within_block_validation.py` | New independent joint SLSQP verifier and machine-readable comparison command |
| `tests/test_within_block_validation.py` | 32 new focused invariance, KKT, stability, and verification tests |
| `results/within_block_validation.json` | Executed teaching-data report with all input rows, full results, comparisons, and individual SLSQP starts |
| `docs/DERIVATIONS.md` | Complete likelihood/unrestricted/restricted derivations retained; clarified proof assumptions, raw-parameter KKT, verification method, and stress scope |
| `docs/MODEL_SPEC.md` | Added audit interfaces, exact feasible-pair retention, and validation scope |
| `docs/RESEARCH_NOTES.md` | Recorded this audit and actual validation results |
| `Readme.md` | Updated instructions, test counts, verification command, and audit link |
| `docs/WITHIN_BLOCK_VALIDATION.md` | This report |

No reference PDF, original statement, observed-data fixture, or unrelated
repository file was changed.

## Mathematical verification

The exact ordered-data likelihood retains C=product of all pre-failure risk
sets and each withdrawal survival factor. Its conditional unrestricted
estimates are b_ic=d_ic/W_i(a). In total/proportion coordinates B_i=b_i1+b_i2
and p_i=b_i1/B_i, each block contributes

    m_i log B_i - B_i W_i(a) + d_i1 log p_i + d_i2 log(1-p_i).

The constraint becomes 0<p_i<=1/2, so strict concavity and the one-sided
derivatives yield p_i^R=min(d_i1/m_i,1/2) and B_i=m_i/W_i(a). Strict
violations pool at b_i1=b_i2=m_i/(2W_i). Already feasible estimates are retained.
Equal observed counts give an active equality with zero order multiplier.

The restricted profile equals the unrestricted profile minus a nonnegative,
count-only constant. The existing strict-concavity/tail proof therefore remains
valid: the unique finite a and every B_i are unchanged whenever an identifying
block with at least two strictly ordered failures exists. The minimum-lifetime
survival, hazard, and median are unchanged. Cause probabilities can change.
The restricted maximized log-likelihood cannot increase.

Zero counts are asymmetric under this labeled order. If d_i1=0, only a b_i1=0
boundary supremum exists. If d_i2=0, the restriction gives a positive equal pair
and can produce attainment even though the unrestricted result was a boundary
supremum. If all blocks have a single failure, the likelihood remains unbounded.
An attained positive order equality is not a zero-parameter boundary: the
existing `interior_mle` status refers to positivity, with active equalities
reported separately. It does not assert strict interiority of the order cone.

No incorrect statistical formula was found in the baseline. The audit corrected
a small floating-point preservation issue: the equal-count branch unnecessarily
recomputed an already feasible pair and could alter it by one rounding step.
It now retains that pair exactly. At active equalities the raw cause scores
need not be zero; the new check explicitly verifies their opposite KKT values.

All assumptions, score equations, KKT multipliers, existence cases, and proof
steps appear in DERIVATIONS.md (L1)-(L8), (M1)-(M8), and (R1)-(R6). The
restriction is imposed on model parameters, not on the observed cause counts;
datasets with d_i1>d_i2 remain valid and require pooling.

## Independent numerical verification

The existing SLSQP tests jointly fit raw b parameters with linear order
constraints and finite-difference gradients. They remain unchanged.

The additional standalone verifier jointly fits log(a/a_scale), log B_i, and
p_i under independent bounds 0<=p_i<=1/2. It uses three fixed feasible starts,
an expanded full likelihood, and an independently checked analytic gradient.
No conditional formula, profile fit, fitted start, or preset active set is
used to obtain its solution. Its time-only coordinate scale changes neither
data nor likelihood. Solver success alone is insufficient: feasibility,
projected stationarity, and numerical-guard checks must also pass.

For the existing teaching data:

| Quantity | Unrestricted | Analytical restricted | Independent SLSQP |
| --- | ---: | ---: | ---: |
| a | 1.173959840838 | 1.173959840838 | 1.173959821550 |
| A b1 | .517708109960 | .388281082470 | .388281083610 |
| A b2 | .258854054980 | .388281082470 | .388281083610 |
| B b1 | .178753429726 | .178753429726 | .178753439640 |
| B b2 | .536260289178 | .536260289178 | .536260336209 |
| Full log-likelihood | 1.070678947337 | .900779910542 | .900779910542 |

Block A has counts (2,1), so it pools; B has counts (1,3), so it stays unchanged.
The cause-1 probabilities change from (2/3,1/4) to (1/2,1/4). Block totals
(.776562164940,.715013718904) and minimum survival at t=.5
(.634625780839,.657914837980) are preserved by the analytical fit.

All three new SLSQP starts converge and pass the independent checks. Maximum
absolute parameter difference from the analytical restricted solution is
4.7031102791e-8; absolute log-likelihood difference is 1.9206858326e-14.
There is no material disagreement within the stated verification tolerances.
The full likelihood includes the same censoring constant in all comparisons.

## Tests and numerical limits

Command: `.venv/bin/python -m pytest -q`.
Actual output: **254 passed in 1.18 seconds**. This comprises 170 original
Stage 1-4 cases, 52 existing restricted cases, and 32 new cases.
Compilation and whitespace checks passed as well.

The new tests verify exact retention of equal pairs, constrained raw KKT,
preservation of totals and the entire minimum survival curve, constant profile
penalties, gradients, standalone reports, solver failures, and extreme times.
Nine time/mark combinations cover small positive times (1e-300 to 1e-60),
times close to one (1-1e-8 through 1-1e-9), and a dataset spanning the smallest
positive float through the largest float below one. Both zero-count directions
are included in the independent comparisons.

The near-one example has a approximately 2.0035e8. It exceeds the original test
oracle's fixed log-a upper guard of 15; this was a validation-range limitation,
not an analytical-estimator failure. The new scaled verifier agrees on it.
Nearly coincident times (.5,.50000001,.50000002), by contrast, require
unrepresentable fitted parameters in the tested design; production correctly
returns `numerical_failure` rather than a false finite MLE.

These are deterministic checks on specified examples, not a simulation study
or proof that every float64 dataset can be fitted. Production still depends on
successful unrestricted profiling; the independent checker is not an automatic
fallback. Cause labels/order require scientific justification. Uncertainty,
novelty, estimator-superiority, and published-table-reproduction claims are not
established here. Across-block and combined restrictions remain unimplemented.

## Reproduction

From the repository root:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.order_restricted_example
.venv/bin/python -m experiments.within_block_validation --output results/within_block_validation.json
```

The JSON contains input rows and all solver diagnostics; the teaching example
prints a shorter interpretation. All data in these commands are the existing
hand-constructed example, not real or randomly simulated observations.
