# Model specification: classical and Bayesian Kumaraswamy BPCS

Current scope: the distribution, BPCS likelihood, both MLEs, both independent
Bayesian estimators and both hierarchical Bayesian estimators are implemented.
The original 60-dataset study has 360 valid fits; the selected appliance
application is also saved. See [simulation completion](SIMULATION_COMPLETION.md)
and the root [README](../README.md) for executed workflows and current results.

The dated design decisions and development milestones below preserve the
project's original reasoning. Statements about deferred experiments refer to
those historical milestones, not the completed study. `References/` retains its
original capitalization.

## Evidence and design decisions

The original project statement from the early README is preserved verbatim in
`PROJECT_STATEMENT.md`. It requires MLEs, independent
or joint Bayesian priors, unrestricted and order-restricted estimation, a full
simulation study, and validation using suitable datasets. It supplies no block
counts, sample sizes, schedules, parameter-sharing rules, cause-observation
rules, restriction direction, or named reliability targets. No separate detailed
project statement was found. All three reference PDFs are available locally.

| Item | Provisional decision explicitly supplied by the user | Literature context |
| --- | --- | --- |
| Blocks | Any configurable k; start later testing with k=2 | Reference 1 — Wang et al. (2023), Section 2.1, Eq. (4), allows k facilities |
| Units/failures | Configurable n_i and m_i, 1 <= m_i <= n_i | Reference 1 — Wang et al. (2023), Section 2.1 |
| Withdrawals | Fixed configurable R_i=(r_i1,...,r_i,m_i), nonnegative integers; n_i=m_i+sum_j r_ij for a completed block | Progressive Type-II experiment within each block; Reference 1 — Wang et al. (2023), Section 2.1 |
| Sharing | One a>0 across blocks and causes; separate b_i1,b_i2>0 for every block | Our Kumaraswamy adaptation; not a stated requirement in the project statement |
| Causes | Exactly two independent latent causes; all recorded causes initially known, coded 1 or 2 | Reference 2 — Wang (2018), Section 2, independence and complete causes; Reference 1 — Wang et al. (2023), Eq. (4) |
| Restrictions | Unrestricted baseline plus implemented provisional b_i1 <= b_i2 independently per block; scientific justification remains pending | Project requires eventual order-restricted analysis; it does not specify an order |
| Targets | Per-block survival, hazard, median time to failure (MTF), and cause-specific failure probabilities | User-selected targets; do not substitute IEED formulas |

Here MTF means **median**, not mean, time to failure.
No research-study n_i, m_i, R_i, true parameter values, or mission times have been
selected. Stage 2 provides a hand-constructed k=2 teaching example with different
block sizes and schedules; its numbers do not fix future experiment designs.
Neither equality of block sizes/schedules nor pooling of b values is assumed.
The implementation accepts any positive k.

## Probability model

For block i=1,...,k and unit u=1,...,n_i, let

\[
T_{iu1}\sim Kw(a,b_{i1}),\qquad T_{iu2}\sim Kw(a,b_{i2}),
\quad X_{iu}=\min(T_{iu1},T_{iu2}),\quad
\delta_{iu}=\arg\min_{c\in\{1,2\}}T_{iuc}.
\]

Conditional on the lifetime parameters, all latent lifetimes are independent
across causes, units, and blocks.
Units are identically distributed within a block, but generally not across
blocks. Shared a is a fixed unknown parameter, not a random effect. It is a
modeling restriction, not a consequence of independence. Both a and b are shape
parameters. There are 1+2k free parameters in this provisional baseline.
The three-parameter model is the k=1 special case.

Support is 0<t<1 on a fixed dimensionless scale. Continuous latent times imply
ties have probability zero. Rounded observed ties require a future measurement
model or documented treatment. No data-dependent maximum scaling is authorized.
If a real dataset needs a transformation, its bounds, units, and likelihood
Jacobian must be justified before fitting.

Write B_i=b_i1+b_i2. Then X_iu ~ Kw(a,B_i), with

\[
S_i(t)=(1-t^a)^{B_i},\qquad
h_i(t)=\frac{aB_i t^{a-1}}{1-t^a},\qquad
\operatorname{MTF}_i=(1-2^{-1/B_i})^{1/a}.
\]

Cause probabilities are P(delta=c)=b_ic/B_i. Cumulative incidence is
P(X<=t,delta=c)=(b_ic/B_i)[1-S_i(t)]. These refer to observed competing failures;
the latent CDF for cause c is different. Stage 1 is implemented per block with
`CompetingRisks(a,b_i1,b_i2)`; its `.minimum` exposes S_i, h_i, and MTF through
`.sf(t)`, `.hazard(t)`, and `.median()`.

## Likelihood structure implemented in Stage 3

Implemented observation fields: `block`, `time`, `cause`, `withdrawals`, accompanied
by explicit block metadata n_i, m_i, R_i. `src/censoring.py` supplies immutable
`BlockScheme`, `Observation`, `BlockData`, and `BPCSExperiment` objects. Completed
blocks validate risk-set transitions, scheme/row agreement, strictly ordered
interior failure times, and fully observed causes. Block IDs must be unique in
the experiment; times may overlap across blocks. See `BPCS_GUIDE.md` for the
step-by-step example and validation contracts.

`Observation.cause=None` reserves an unknown-cause representation, but current
`BlockData` validation rejects it. Missing causes are not observations of a third
cause. A missingness model must be specified before they enter a likelihood.

At a known cause-c failure, the joint time/cause density is
g_ic(t)=f_ic(t) S_i,other(t). A surviving withdrawn unit contributes S_i(t):
its exact failure time is unknown, but it is known to exceed t. Under uniform
random, noninformative removal of survivors, each row therefore contributes
g_i,cause(time) times S_i(time) raised to its withdrawal count. A block multiplies
these contributions with its parameter-independent progressive-ordering factor;
independent blocks multiply. Stage 3 derives and implements the exact ordered-data
likelihood and log-likelihood in `src/likelihood.py`. Its block constant is
C_i=product_j Y_ij, where Y_ij is the pre-failure risk-set size. Full constants
are included by default; `include_constant=False` removes only their logarithm.
See DERIVATIONS.md, (L1)-(L8), for the complete derivation, event convention,
time-only versus parameter-dependent terms, numerical contracts, and example.
Stage 4 derives all scores and implements profile and independent direct MLEs
in `src/mle.py`; see DERIVATIONS.md, (M1)-(M8). The model's parameter domain
remains strictly positive. An estimate reported as zero for a zero observed
cause count is a closure limit with status `boundary_supremum`, not a change to
the latent distribution assumptions or a positive-domain MLE. If every block
has only one observed failure, the likelihood is unbounded and no finite
parameter vector is reported. The finite maximum is unique when some block
has at least two strictly ordered failures, with interior attainment requiring
positive counts for both causes in every block. Numerical failure remains
distinct from these mathematical classifications.

There is no single pooled-population survival curve without specifying a target
population and block weights. A weighted mean of b parameters does not generally
produce the survival mixture over facilities. Facility contrasts and any pooled
target remain to be specified later.

## Differences from the references

- **Reference 1 — Wang et al. (2023):** *Inference for block progressive censored
  competing risks data from an inverted exponentiated exponential model.*
  Local file: `../References/Reference_1.pdf`; DOI 10.1002/qre.3382.
  Its Section 2.1 and Eq. (4) supply the block design and observation organization;
  Eqs. (5)-(7) describe IEED with facility-specific theta_is and shared lambda.
  Our b_ic and a sharing structure is analogous but uses a different distribution,
  support, and mathematical formulas. Its Eq. (8) requires care; see research notes.
  Its weighted parameter summaries and hierarchical model are not adopted.
- **Reference 2 — Wang (2018):** *Inference of progressively censored competing
  risks data from Kumaraswamy distributions.* Local file:
  `../References/Reference_2.pdf`; DOI 10.1016/j.cam.2018.05.013.
  Eq. (1), printed p.720 (PDF p.2), defines Kw(alpha,beta), with a=alpha,
  b=beta here. Eq. (3), printed p.721, gives independent cause and survival
  contributions for a single progressive Type-II block. Section 3.2 includes
  alpha_1=alpha_2; general results also permit unequal alpha values. Neither the
  full block design nor global sharing across facilities follows from that paper.
- **Reference 3 — Mahto et al. (2022):** *Inference for partially observed
  competing risks model for Kumaraswamy distribution under generalized progressive
  hybrid censoring.* Local file: `../References/Reference_3.pdf`;
  DOI 10.1080/02664763.2021.1889999. Eqs. (1)-(2), printed p.2066 (PDF p.3),
  use common alpha and cause parameters lambda_j. Sections 2.1 and 4 address
  missing causes, hybrid stopping, and restrictions. We do not adopt its hybrid
  stopping rule or missingness probability. Its Section 4.1, Theorems 4.1-4.2
  (PDF pp.11-12), provides the order/pooling framework now adapted and derived
  for our completed multi-block model; the unrestricted baseline stays available.

These are supporting structures, not proof that our adapted BPCS model or its
MLEs have any existence, uniqueness, or asymptotic properties without derivation.

## Provisional within-block order-restricted MLE (2026-10-26)

The user specifies b_i1<=b_i2 independently for every block. Positivity, common
a, all censoring schemes, known causes, and the exact full likelihood are
unchanged. No cross-block ordering is added. The order means cause-1 hazard is
no larger, its eventual failure probability is at most 1/2, and its latent
lifetime is stochastically larger. These implications require a scientific
justification for the cause labels and direction; this remains provisional.

With B_i=b_i1+b_i2 and p_i=b_i1/B_i, the likelihood separates into time/total
and cause-proportion terms. At fixed a, B_i=m_i/W_i(a) and
p_i^R=min(d_i1/m_i,1/2). Violating blocks pool their cause parameters at B_i/2;
nonviolating pairs stay unchanged. The restricted profile differs from the
unrestricted profile by a nonnegative constant depending only on counts.
Therefore the shared a and all block totals B_i remain unchanged. The full
restricted log-likelihood cannot exceed the unrestricted value. See
DERIVATIONS.md (R1)-(R6) for proofs, literature provenance, and qualifications.

`src/order_restricted.py::fit_order_restricted_mle` returns MLEResult, reuses the
existing profile solver, and checks the unchanged likelihood. Diagnostics
separate active equalities, strictly pooled blocks, and zero parameters.
`interior_mle` means positive-parameter attainment even when equality is active.
If d_i2=0, the restricted pair is positive and equal; unlike the unrestricted
fit, this does not create a zero boundary. If any d_i1=0, b_i1=0 is reported
only as a closure limit (`boundary_supremum`). These classifications assume
some block has at least two strictly ordered failures; otherwise the likelihood
is unbounded regardless of the order, with no finite estimate returned.
Numerical/optimization failure remains separate from these mathematical cases.

The two-block teaching example and independent SLSQP checks are available in
`experiments/order_restricted_example.py` and `tests/test_order_restricted.py`.
The completed audit adds `experiments/within_block_validation.py` for a
standalone, jointly constrained SLSQP verification and JSON report, plus
`tests/test_within_block_validation.py` for invariance and extreme-time tests.
The production fit retains all already-feasible pairs exactly, including equal
counts, and checks relative preservation of each block total. Within-block
order equalities do not add any ordering between facilities. See
WITHIN_BLOCK_VALIDATION.md for the inspected files, numerical results, and scope.
That MLE-only milestone did not implement uncertainty. The classical extension
below now supplies the supported classical uncertainty procedures. The Bayesian
extension is specified at the end of this document; simulation studies and
real-data analysis remain unimplemented.

## Classical uncertainty extension (2026-10-26)

The classical inference module now supplements the preceding MLE milestone;
its earlier scope statements describe that milestone historically. All model,
data and censoring assumptions above remain unchanged. With B_i=b_i1+b_i2,
p_i=b_i1/B_i, the likelihood factors into independent parameter components:
a time part in (a,B_1,...,B_k) and a binomial cause-label part for each p_i.
This factorization assumes fixed m_i and noninformative withdrawals with fully
known causes. See DERIVATIONS.md I1–I7 for complete first/second derivatives,
analytical observed information, transformations and numerical checks.

Supported procedures in `src/uncertainty.py`:

- Unrestricted interior MLE: full observed-information covariance, SEs for
  a,B,p,b1,b2, marginal normal/log/logit Wald intervals as appropriate.
- Time component (a,B): observed-information covariance and approximate marginal
  intervals also under within-block restriction and zero-cause closure limits,
  subject to regular time-model asymptotics and numerical information checks.
- Cause probabilities: exact marginal Clopper–Pearson sets; within-block
  restriction intersects the cause-1 set with [0,1/2]. The set can be empty,
  and remains explicitly empty. Cause-2 sets are complements.
- Block survival, hazard and median lifetime (MTF means median here): shared
  Stage 1 point values and pointwise delta SE/transformed intervals from the
  time covariance. Restricting p leaves their estimates and uncertainty unchanged.
- Cause-specific cumulative incidence: fitted values for both models; delta
  uncertainty only for regular unrestricted interior fits.

Unsupported: ordinary full covariance or b/probability Wald SEs at a zero
cause boundary; ordinary restricted p SEs; restricted individual-b or incidence
confidence intervals. Active positive equality is distinguished from zero
parameters and from inactive constraints. No uniform coverage claim is made
for normal/delta approximations; exact probability coverage is marginal under
the binomial model and correct order. Intervals are not simultaneous bands.
Ill-conditioned/nonpositive information fails without regularization; failed,
unbounded or mismatched fits are rejected. No epsilon replaces a boundary.
The full observed information is not the expected Fisher information.
Published numerical-table reproduction, simulations and empirical coverage
studies remain separate pending work. No Bayesian or additional ordering
methods are included by this extension.


The final numerical audit preserves these assumptions and public interfaces.
Subnormal distribution tails are evaluated on the log scale; covariance
symmetrization avoids intermediate overflow; nonrepresentable Wald endpoints
are explicitly unavailable. These arithmetic fixes do not extend statistical
inference at nonregular boundaries. See FINAL_AUDIT.md for verification.


## Bayesian prior specifications (2026-11-01)

The lifetime model, support, fixed schedules and known-cause assumption do not
change. A prior on the single common unknown a does not introduce unit-specific
frailty or a different a for each facility. Blocks remain independent conditional
on their lifetime parameters; shared priors induce marginal dependence.

Implemented cases are unrestricted and b_i1<=b_i2 for every block, under:

1. Independent scale prior pi(a,b) proportional to 1/a times product(1/b_is).
   Both observed cause counts must be positive in every block, M>1 and D>0.
   Restricted zero-count cases are conservatively unsupported, not all declared
   mathematically improper. The a marginal and Gamma totals are unchanged by
   this within-block restriction; Beta cause proportions are truncated at 1/2.
2. Hierarchical b_is|kappa_s,lambda_s~Gamma(kappa_s,lambda_s), with explicit
   proper Gamma priors for a, each kappa_s and each lambda_s. All Gamma laws use
   shape-rate. Hyperpriors are configuration inputs. The restricted model
   conditions the entire joint posterior on within-block order; it does not
   independently normalize conditional prior truncations. Its a marginal and
   totals generally change under restriction.

The posterior-mean and variance conditions, verified conditional distributions,
Jacobian factors and credible-interval qualifications are in
BAYESIAN_DERIVATIONS.md. `src/bayesian.py::fit` returns structured draws,
posterior-mean estimates where justified, intervals, moments, diagnostics,
reliability summaries, warnings and reproducibility controls. Infinite/unverified
means are None; improper priors, numerical failure and failed diagnostics have
separate classifications. Classical likelihood/MLE/uncertainty source files
remain unchanged. No across-block order, missing causes, hybrid censoring,
new loss, simulation study or real-data inference is introduced.
