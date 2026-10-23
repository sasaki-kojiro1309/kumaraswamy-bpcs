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
