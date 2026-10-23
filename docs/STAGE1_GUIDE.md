# Stage 1 function guide and verified checks

This guide explains the existing implementation in `../src/model.py`.
For formulas and derivations see `DERIVATIONS.md`, (D1)-(D7).

## Statistical meaning of each public function

| API | Plain-language interpretation | Why it is needed |
| --- | --- | --- |
| `Kumaraswamy(a,b)` | A lifetime model on (0,1) with two positive shape parameters | Describes a latent cause, or the observed minimum with b=b1+b2 |
| `.pdf(t)` | Density of lifetime near t; f(t)dt approximates a short-interval failure probability | Describes exact failure times; density itself is not a point probability |
| `.cdf(t)` | P(T<=t), the fraction failing by t | Cumulative failure probability |
| `.sf(t)` | P(T>t), the fraction surviving beyond t | Reliability and information supplied by a withdrawn survivor |
| `.hazard(t)` | Instantaneous failure rate among those still surviving at t | Measures the changing conditional rate of failure; can exceed one |
| `.median()` | Time at which half the lifetimes have ended | Median time to failure (MTF), not the mean |
| `.logpdf(t)` | Logarithm of the density | Retains numerical information for extremely small densities |
| `.logsf(t)` | Logarithm of survival | Retains information when ordinary survival underflows to zero |
| `CompetingRisks(a,b1,b2)` | One block's two independent latent lifetimes with shared a | Keeps the competing-risk assumptions explicit |
| `.minimum` | Distribution of the observed time X=min(T1,T2) | Its pdf/cdf/sf/hazard/median describe what can be observed for the unit |
| `.cause_pdf(t,c)` | Joint time-and-cause density: g_c(t)dt approximates P(t<X<=t+dt,delta=c) | Describes a known-cause failure; integrates to its cause probability |
| `.cause_logpdf(t,c)` | Logarithm of that joint density | Numerically stable counterpart |
| `.cause_probability(c)` | Probability cause c eventually wins the competition | P(delta=c)=b_c/(b1+b2) |
| `.cause_cdf(t,c)` | Probability of failure by t from cause c | Cumulative incidence, P(X<=t,delta=c) |

The latent cause CDF P(T_c<=t) differs from cumulative incidence: another cause
can fail first and prevent that latent lifetime from being observed.

For example, choose a=2,b1=1,b2=2 for illustration. At t=.5, the observed
minimum's survival is .421875 and its hazard is 4. The eventual cause-1
probability is 1/3; the cause-1 probability by .5 is .1927083333. The observed
median is approximately .454202019. These are formula evaluations with chosen
parameters, not estimates, simulations, or published numerical results.

## Which Stage 1 tests passed?

On 2026-10-23, before Stage 2 changes, the original suite was rerun:
**20 passed in 0.34 seconds** (`python -m pytest -q` in the project environment).
Parametrization accounts for multiple cases per named test.

| Test family in `../tests/test_model.py` | Cases | What it establishes numerically |
| --- | ---: | --- |
| Density integral and transformed Beta distribution | 5 | Density integrates to 1; T^a follows Beta(1,b), including singular-density cases |
| Calculus, survival, hazard, median | 3 | F'=f, -(log S)'=h, F+S=1, increasing F, median divides probability in half |
| Independent competing-risk construction | 3 | Product survival, correct joint cause densities, sum of densities/hazards, cause integrals, cumulative incidence, minimum median |
| Uniform distribution and support conventions | 1 | Known analytical case, scalar/array outputs, zero endpoint density convention |
| Invalid parameters | 6 | Nonpositive, nonfinite, or nonscalar parameters rejected |
| Invalid times, causes, combined shape | 1 | NaNs and illegal cause codes rejected; hazard requires interior times |
| Numerical tails | 1 | Tiny nonzero CDF retained; log functions and direct hazard survive ordinary-probability underflow |

These tests validate the distribution layer. They do not establish MLE properties
or certify an experimental censoring design. Baseline formulas come from
**Reference 2 — Wang (2018)**, Eq. (1), with competing-risk structure in Section 2,
Eq. (3). The minimum and cause-probability identities are derived explicitly in
DERIVATIONS.md rather than assumed from an IEED model.
