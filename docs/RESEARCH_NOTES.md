# Research log

## 2026-10-23: first milestone

Read the original project statement (now preserved verbatim in `PROJECT_STATEMENT.md`). Found all three reference PDFs
in `References/` (29, 18, and 29 pages respectively). Extracted all PDFs for
inspection; read the model/design and relevant mathematics sections for Stage 1
and visually checked key source pages. This is not an audit of every theorem,
simulation table, or Bayesian derivation in the papers. No external paper was
needed to establish Stage 1. The source PDFs were preserved.

The project statement leaves design details open. The user explicitly supplied
the provisional choices recorded in MODEL_SPEC.md during this milestone. They
are not presented as requirements stated in the papers.

## Source map

| Label | Local source | Relevant source locations |
| --- | --- | --- |
| Reference 1 — Wang et al. (2023) | `../References/Reference_1.pdf` | Section 2.1, PDF p.6 / printed p.2741, Eq. (4): blocks and observations; Eqs. (5)-(7): IEED block parameters. Section 2.2, PDF p.11 / printed p.2746, Eqs. (8)-(10): likelihood structure. |
| Reference 2 — Wang (2018) | `../References/Reference_2.pdf` | Eq. (1), PDF p.2 / printed p.720: Kw CDF/PDF. Section 2, Eqs. (2)-(3), PDF pp.2-3: progressive Type-II cause/survival contributions. Section 3.2, PDF p.4: common-alpha special case. |
| Reference 3 — Mahto et al. (2022) | `../References/Reference_3.pdf` | Eqs. (1)-(2), PDF p.3 / printed p.2066: Kw with shared alpha. Section 2.1, PDF pp.4-6: hybrid stopping and partial causes; Eq. (6) is not the baseline BPCS likelihood. |

## Discrepancies requiring care

In **Reference 1 — Wang et al. (2023)**, Eq. (8), printed p.2746 / PDF p.11,
the displayed cause-2 contribution contains f_2(x)[1-S_1(x)]. The symbol S was
previously defined as survival, so the independent competing-risk contribution
must instead be f_2(x)S_1(x). The withdrawal expression also displays a generic
t where the observation time x_ij is needed. These were checked in a rendered
page, not inferred solely from extracted text. Treat them as apparent printed
notation errors; our event-based derivation is the basis of implementation.
Do not claim this observation invalidates all subsequent paper results.

The expanded Eq. (9) omits time-only factors (such as products of x_ij^-2);
these can be ignored for parameter maximization but require care when claiming
an exact sampling density or comparing absolute likelihood values later.

**Reference 3 — Mahto et al. (2022)**, printed p.2066 / PDF p.3, contains
descriptive shape/affine-transformation statements that should not be copied
without verification. For example, a>1,b>1 gives a density vanishing at both
ends and an interior mode, not a decreasing density; the mapping from (0,1) to
(L,U) is L+(U-L)X. Stage 1 relies on the displayed CDF/PDF and independent
calculus, not those descriptive statements. No real-data transformation has
been selected or applied.

## Scientific status

1. Established literature: Kumaraswamy CDF/PDF, independent competing-risk
   contributions, progressive Type-II organization within facilities.
2. Our implementation/validation: Stage 1 distribution layer, derivations,
   numerical stability choices, mathematical tests.
3. Adaptation awaiting later validation: common-a, facility-specific-b
   Kumaraswamy model under BPCS. No likelihood code yet.
4. Novelty: no novelty claim. Literature searches and further derivations would
   be required before identifying a methodological contribution.

No simulations, fitted real data, MLEs, uncertainty estimates, or paper-table
reproductions are reported. Bayesian and order-restricted modules are explicit
placeholders. The original deleted tracked `Reference_Paper.pdf` shown by Git
was already absent at inspection; it has not been restored or modified.

## Next milestone recorded after Stage 1 (historical)

Stage 2: formalize block metadata and observation schema; validate integer
counts, within-block strict ordering, causes, risk sets, and completed-block
n_i=m_i+sum_j r_ij. Retain extensibility for unknown causes but reject them in
the known-cause baseline until a missingness model is authorized. Stage 3 will
then derive the full block likelihood from the validated design.

## Stage 1 validation record

On 2026-10-23, `.venv/bin/python -m pytest -q` completed with **20 passed**
in 13.79 seconds. Environment: Python 3.12 on macOS arm64; NumPy 2.5.4,
SciPy 1.18.1, pytest 9.1.1. Exact installed versions, including the reserved
pandas/Matplotlib stack, are recorded in `../requirements-lock.txt`.
Compilation checks and `git diff --check` also passed. The preserved project
statement was compared byte-for-byte with the original Git version.

The tiny-probability test uses relative tolerance 1e-13 with zero absolute
tolerance: at t=1e-100, evaluation through log/exp has relative roundoff on the
order of |log(t)| times machine precision. This still detects a naive CDF that
incorrectly rounds the nonzero failure probability to zero.

These are mathematical software checks, not experimental performance findings
or evidence of MLE existence/uniqueness. Stage 2 remains unstarted.

## 2026-10-23: Stage 2 completed

The user requested a function-by-function Stage 1 explanation, then authorized
Stage 2 without likelihood construction. Reran the unchanged Stage 1 suite:
20 passed in 0.34 seconds. Saved its function meanings and test families in
STAGE1_GUIDE.md. Re-read **Reference 1 — Wang et al. (2023)**, Section 2.1,
Eq. (4), printed p.2741 / PDF p.6, for the block experiment and observation
structure. The Kumaraswamy support replaces IEED support; no IEED distribution
formula enters the implementation.

Implemented immutable scheme, observation, completed-block, and experiment
objects in `src/censoring.py`. Constructors validate counts, risk transitions,
block identities, exact row-to-scheme matches, time support/order, and known
causes. Exports use pandas. Parameters remain separate from data. Single-cause
blocks remain valid even though future MLEs may require boundary handling.
Missing cause None is representable as a row but not admitted to baseline data.

Ran `.venv/bin/python -m pytest -q`: **75 passed in 11.58 seconds**, consisting
of 20 Stage 1 and 55 Stage 2 cases. One Stage 2 test exhaustively checks all
63 valid completed schedules with n<=6 against the risk-set formula. Other
tests reject malformed input and verify completed, Type-II, and one-failure
designs, block separation, immutable data, and exported-table isolation.
`compileall` and `git diff --check` passed as well. No new dependency was needed.

Ran `.venv/bin/python -m experiments.stage2_example`; its output matches the
hand-calculated risk table in BPCS_GUIDE.md. Block A: n=6,m=3,R=(1,0,2),
risk sets (6,4,3); block B: n=7,m=4,R=(0,1,0,2), risk sets (7,6,4,3).
Each ends with zero units under test; total 13=7 failures+6 withdrawals.
The rows in `data/examples/two_blocks.json` are explicitly hand-constructed
teaching data. The script's chosen model parameters only demonstrate the
existing reliability functions. They are not fitted values or simulation results.

Structural validation cannot establish random survivor selection or a correct
distributional model. These remain experimental/model assumptions. Likelihood,
MLE, random censoring simulation, Bayesian estimation, and order restrictions
remain unimplemented. **Stage 3 is the next milestone and has not started.**

## 2026-10-23: Stage 3 completed

Re-read **Reference 1 — Wang et al. (2023)**, Section 2.1, Eq. (4) (PDF p.6),
and Section 2.2, Eqs. (8)-(9) (PDF p.11), for observation organization and the
block product. Re-read **Reference 2 — Wang (2018)**, Eqs. (1)-(3) (PDF pp.2-3),
for the Kumaraswamy functions, cause-specific contributions, and single-block
progressive-ordering constant. Maintained the existing caution about the
apparent survival-factor error in the displayed Eq. (8) of the former.

Derived the row event contributions, exact constant via conditional risk sets,
block and joint likelihoods, and full log-likelihood in DERIVATIONS.md (L1)-(L8).
Specified the sample space as ordered times with their observed cause sequence,
without unit identities, conditional on fixed completed schemes. Uniform random
noninformative removal is assumed. No extra multinomial factor applies to the
observed cause sequence. The censoring constant is product_ij Y_ij, distinct
from any time-only terms. Both are retained by default; omitting the censoring
constant removes only log C.

Added `src/likelihood.py`, `tests/test_likelihood.py`, and
`experiments/stage3_example.py`. Model and censoring source files, the original
Stage 1/2 test files, and Stage 2 data/example were left unchanged. The joint API
takes a shared a and exact block-ID-to-(b1,b2) mapping, so parameter vectors and
priors can be handled by later estimator adapters without duplicating likelihood
mathematics. Full constants default to included. Nonfinite log arithmetic is
reported separately from invalid parameter input; ordinary likelihood underflow
and overflow remain possible and documented.

Incremental validation: `.venv/bin/python -m pytest -q tests/test_likelihood.py`
gave **56 passed in 0.59 seconds**. The complete suite then gave
**131 passed in 0.85 seconds** (20 Stage 1, 55 Stage 2, 56 Stage 3).
`compileall` and `git diff --check` passed. Tests cover hand calculations,
closed-form and sequential conditional formulations, latent contributions,
summation over causes, normalized ordered-time densities, first-failure cause
probabilities, exact constants, parameter/mapping validity, and numerical tails.
No new dependencies were required.

Executed `.venv/bin/python -m experiments.stage3_example` on the unchanged
teaching data, at chosen a=2, b_A=(1,2), b_B=(.8,1.2). C_A=72, C_B=504,
and C=36288. The joint kernel log-likelihood is -13.332370141561 and the full
log-likelihood is -2.833127754473, differing by log C=10.499242387087. These are
evaluations at chosen parameters, not maximized likelihoods or fitted estimates.
The per-row contribution table is printed by the script and the block-level
results are recorded in DERIVATIONS.md.

**Stage 4 is not started.** It will need scores for shared a and block-specific
b values, conditional-estimation/profile analysis, treatment of zero-cause
counts and other boundary behavior, positive parameterization, numerical starts
and convergence checks, and comparison to direct optimization. No MLE,
Bayesian estimation, order-restricted method, simulator, or uncertainty procedure
was implemented in Stage 3. Published numerical MLE validation remains pending.

## 2026-10-23: Stage 4 completed

Read **Reference 1 — Wang et al. (2023)**, Section 3.1, Theorem 1 and the profile
construction preceding Theorem 2 (PDF pp.13-14), and **Reference 2 — Wang (2018)**,
Section 3.2, Theorems 3-4 and Appendices C-D (PDF pp.4,16-17). Derived scores
and conditional estimates from our own (L8), with one a score summing all
facilities. No IEED formula or unrestricted unequal-alpha formula is transferred.

Added `src/mle.py`, `tests/test_mle.py`, and `experiments/stage4_example.py`.
Stage 1-3 model/data/likelihood source and tests, teaching data, and previous
example scripts were left unchanged. No additional dependency was needed.

The b scores give b_ic=d_ic/W_i(a) for positive counts. Zero counts yield a
finite closure supremum at b_ic=0, not an interior MLE; survival contributions
still penalize positive b_ic. Derived a strictly concave profile in a by writing
W_i as a positive mixture of exponentials and examining its second derivative.
Tail limits establish a unique finite maximum when at least one block has m_i>=2.
When all blocks have m_i=1 the profile is unbounded at a tending to infinity.
These qualifications were derived for our model, not assumed from the papers.

Profile estimation uses an adaptive sign bracket and Brent root solving in
log(a). Independent direct joint optimization uses multistart L-BFGS-B with
log parameters and no profile-based warm starts or conditional updates. Both
report solver details, active score checks, exact boundary limits, and a
profile/joint objective consistency check. One-sided boundary scores are
reported separately. Numerical guards/failures do not masquerade as MLEs.
No uncertainty interpretation is attached to the optimizer's internal curvature.

The Stage 4 tests passed **39 cases**. The full command
`.venv/bin/python -m pytest -q` reported **170 passed in 1.11 seconds**.
Compilation and whitespace checks passed. Tests include independent finite
differences of Stage 3, independent inner numerical maximization, profile/direct
agreement, distant initial values, arbitrary block counts, single-cause and
single-failure cases, boundary limits, budget exhaustion, numerical failures,
invalid controls, and deterministic known-parameter recovery.

Synthetic recovery fixtures use fixed population-quantile grids, not random
samples or a simulation generator. Complete schemes R=0 are valid BPCS special
cases. The tests recover three a values and two block-specific b pairs within
declared numerical/finite-grid tolerances. This does not establish Monte Carlo
bias, MSE, consistency, coverage, or a complete experimental-process simulation.

Executed `.venv/bin/python -m experiments.stage4_example` on the existing data.
Profile a=1.173959840838, b_A=(.517708109960,.258854054980),
b_B=(.178753429726,.536260289178); full log-likelihood=1.070678947337.
The independent direct log-likelihood differs by about 8.9e-16. Both are interior
fits; the largest active log-parameter scores are approximately 4.4e-16 and
6.1e-9. Cause-1 fitted probabilities equal observed proportions 2/3 and 1/4.
These are fits to hand-constructed teaching data, not published results.

**Stage 5 remains unstarted.** Observed information, standard errors, confidence
intervals, profile intervals, Bayesian estimation, and random simulations were
not implemented. Published numerical baseline reproduction remains pending
matching source assumptions and data; no paper table is claimed reproduced.

## 2026-10-26: within-block order-restricted MLE

Scope explicitly requested by the user: derive, implement, test, and document
only b_i1<=b_i2 independently in every block. The direction is provisional;
no cross-block order or censoring/model assumption changed. No Bayesian method,
uncertainty method, simulation study, or real-data analysis was added.

Reviewed MODEL_SPEC, DERIVATIONS, these notes, model/censoring/likelihood/mle
source, and all three local references' relevant model and estimation sections.
Baseline `.venv/bin/python -m pytest -q`: **170 passed in 1.24 seconds**.
No Stage 1-4 source/test rewrites or dependency changes were necessary.
The pre-existing unrelated untracked `molecular-property-research/` was untouched.

Literature: Reference 1, Section 2.1 / Eq. (4), supplies block organization but
uses IEED. Reference 2, Section 3.2, Theorems 3-4 (PDF p.4 / printed p.722),
supplies the common-alpha single-block Kumaraswamy baseline. Reference 3,
Section 4.1, Theorems 4.1-4.2 (PDF pp.11-12 / printed pp.2074-2075), supplies
the single-experiment order/pooling framework and unchanged alpha result.
Those two pages were also rendered and visually checked. Theorem 4.1's first
branch appears to label its left side lambda_1 while its right side and
j=1,2 condition use lambda_j; the following page gives the latter consistently.
Our formulas are independently derived from (L8), not copied from that typo.
The paper's partially observed causes and generalized progressive hybrid
stopping are not adopted. Zero-count treatment below extends our own boundary
analysis beyond the positive-count hypotheses stated in those theorems.

Derivation (R1)-(R6) separates B_i=b_i1+b_i2 and p_i=b_i1/B_i. For each fixed a,
B_i=m_i/W_i(a) uniquely, and p_i^R=min(d_i1/m_i,1/2), with p=0 only as a
closure limit. When the order is violated both b parameters pool at B_i/2.
The constrained profile is the unrestricted profile minus a nonnegative
count-only constant, so a and B_i are unchanged. Finite attained/supremum
log-likelihood ordering follows, with strict loss precisely when some block's
cause-1 count exceeds cause-2 count. No novelty or superiority claim is made.

Interpretation: cause-1 hazard is no larger and its latent lifetime is
stochastically larger, within every block; eventual cause-1 probability is at
most 1/2. Domain knowledge must justify these labeled-cause statements. It is
not a cross-facility order or an empirical assertion inferred from these data.

Boundary distinction: if d_i2=0, the restricted optimum has two positive equal
parameters, even though the unrestricted estimator had a zero-cause-2 limit.
If d_i1=0, only a zero-b_i1 boundary supremum exists. Finite attainment for the
restricted fit needs d_i1>0 in every block, together with at least one block
having two strictly ordered failures. If all blocks have one failure the
likelihood stays unbounded. Equality-count blocks have an active equality and
zero KKT multiplier. These are mathematically different cases, not solver
statuses inferred from a parameter's proximity to a numerical bound.

Implementation: completed src/order_restricted.py with fit_order_restricted_mle
and a narrowly validated compatibility fit entry point. Reuses fit_mle, its
result type, stable cached summaries and gradient. The unchanged Stage 3
likelihood evaluates positive fits. Actual zero-b1 cases use the exact linear
boundary identity; no epsilon estimate is substituted. Diagnostic checks cover
order residuals, total/a scores, p stationarity with upper/lower KKT multipliers,
likelihood penalty and profile/joint agreement. Active equalities, strictly
pooled blocks, and zero parameters are recorded separately. Failures propagate
without certifying a fit. `interior_mle` retains its positive-parameter meaning
and can include an active positive order equality.

Added tests/test_order_restricted.py: **52 passing cases** including separate
joint SLSQP fits with raw nonnegative b parameters, linear order constraints,
finite-difference gradients, an independent row-expanded objective, and two
fixed feasible starts. No analytical conditional update or profile fit supplies
the numerical oracle's estimates or initialization. Checks include multiple
block counts, both zero-count directions, equality counts, inactive constraints,
likelihood ordering, totals/common-a invariance, unbounded likelihood, invalid
inputs, exhausted budgets, injected evaluation failure/objective mismatch,
and no mutation of the unrestricted fit. No random samples or large Monte
Carlo experiment were generated.

Added experiments/order_restricted_example.py using unchanged two_blocks.json.
Both fits converge with status interior_mle. Shared a=1.173959840838;
unrestricted A=(.517708109960,.258854054980), restricted
A=(.388281082470,.388281082470). B is unchanged at
(.178753429726,.536260289178). A's equality is active, B's is inactive,
with no zero boundaries. Full log L changes from 1.070678947337 to
.900779910542; the penalty is .169899036795. These are hand-constructed
teaching data, not a published reproduction or performance study.

Validation: `.venv/bin/python -m pytest -q` reported **222 passed in 1.19
seconds**. All 170 existing cases remain passing. The reproducible example,
`python -m compileall -q src tests experiments`, and `git diff --check` passed.
Independent numerical tests agree in parameters within declared tolerances
and in maximized log-likelihood within 1e-9.

Future priorities: (1) justify cause labels/order scientifically; (2) broaden
focused deterministic stress checks for float64 range, nearly coincident times,
and large k, including possible restricted-only fitting when the unrestricted
fit is unrepresentable; (3) derive other justified restriction types only under
separately approved scope. No unresolved mathematical contradiction was found
for the current data contract. Numerical finite range and limited oracle
fixtures remain limitations. Stage 5 uncertainty and later research remain
unstarted; this milestone stops at the restricted MLE baseline.

Final teaching-data independent SLSQP check: a=1.173959807259,
A=(.388281063179,.388281063179), B=(.178753408926,.536260247045).
Maximum absolute difference from analytical fitted parameters was 4.22e-8;
absolute full log-likelihood difference was 1.32e-14 (rounded upward).
Final full-suite rerun: **222 passed in 1.10 seconds**.

## 2026-10-26: completion audit and independent numerical stress verification

Read the attached completion request, inspected the existing modules, derivations,
model specification, tests, and examples. The working tree already contained a
correct within-block estimator and 52 restricted tests, with 222 total passing
cases. Preserved the unrestricted implementation and all existing tests. A
SHA-256 comparison confirmed `src/model.py`, `src/censoring.py`,
`src/likelihood.py`, `src/mle.py`, and the five existing test files are unchanged
from the beginning of this audit. Existing user changes were not reset.

Rechecked the exact likelihood, d_ic/W_i conditional estimates, unrestricted
profile, and the restricted B,p separation. The strict-concavity proof remains
valid; added the uniform-convergence justification for differentiating its
exponential series. Added raw-parameter KKT equations at positive order
equalities, complementing the existing p-coordinate multipliers. No erroneous
statistical formula was found in the baseline.

Made one small production correction: equal-count pairs that already satisfy
the constraint were unnecessarily re-exponentiated, which can change one
floating-point rounding step. They now remain bit-for-bit unchanged; only
strict violations pool. Production also explicitly checks relative block-total
preservation and exposes those residuals in diagnostics.

Added `experiments/within_block_validation.py`. Its SLSQP optimization jointly
fits scaled log a, log totals, and bounded proportions, with an expanded full
likelihood and three fixed feasible starts. It does not call either analytical
estimator to obtain its estimates or initialization. Its independent gradient
is checked by finite differences; each result requires solver success,
feasibility, projected stationarity, and no guard contact. The old raw-b
finite-difference oracle and its tests remain unchanged.

Found a verification-range limitation in the old oracle: its log-a guard of
15 excludes a valid near-one-time example whose estimated a is about 2.0035e8.
The new data-scaled coordinate allows independent verification of that case.
It also agrees for very small times and for a dataset spanning the representable
support endpoints, including both zero-count directions. Near-coincident times
with unrepresentable estimates still return an explicit numerical failure.
No automatic fallback or arbitrary-precision estimator was introduced.

Added 32 cases in `tests/test_within_block_validation.py`. Full validation:
`.venv/bin/python -m pytest -q` reported **254 passed in 1.18 seconds**.
Compilation and whitespace checks passed. The expanded teaching example prints
the original seven rows, three fits, pooling decisions, cause probabilities,
and survival invariance. The report command writes the executed comparison to
`results/within_block_validation.json`, including all starts and input rows.

Teaching-data SLSQP: a=1.173959821550, A=(.388281083610,.388281083610),
B=(.178753439640,.536260336209), log L=.900779910542. All three starts pass
checks. Maximum absolute parameter difference from the analytical restricted
fit is about 4.7032e-8 and log-likelihood difference is about 1.9207e-14.
No material disagreement remains within the stated tolerances. Common a,
totals, and minimum survival are preserved by the analytical restriction; only
cause probabilities change in violating blocks.

See `docs/WITHIN_BLOCK_VALIDATION.md` for the file inventory, mathematical
audit, exact commands, result table, and remaining limits. The within-block
stage is complete for the provisional model and documented numerical domain.
Across-block PAVA, combined restrictions, Bayesian estimation, uncertainty,
and simulation studies remain outside this task and were not implemented.
