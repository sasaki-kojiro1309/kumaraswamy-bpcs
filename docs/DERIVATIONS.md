# Distribution, likelihood, and MLE derivations (Stages 1, 3, and 4)

These are our derivations and validation identities, not copied BPCS estimators.
Reference labels below retain the project naming convention. All formulas here
use a,b>0 and 0<t<1. Both parameters are shapes.

## One latent lifetime

**Reference 2 — Wang (2018)**, Eq. (1), printed p.720 (PDF p.2), gives
the Kumaraswamy CDF. Our a,b equal its alpha,beta. Consequently,

\[
F(t)=1-(1-t^a)^b,\qquad S(t)=P(T>t)=1-F(t)=(1-t^a)^b. \tag{D1}
\]

The CDF answers "what fraction has failed by t?" Survival answers "what fraction
lasts beyond t?" A withdrawal at t supplies precisely the latter information.
Differentiating the CDF by the chain rule gives

\[
f(t)=F'(t)=-b(1-t^a)^{b-1}(-at^{a-1})
=ab t^{a-1}(1-t^a)^{b-1}. \tag{D2}
\]

For a small dt, f(t)dt approximates the chance of failure in (t,t+dt).
It is a density, so its value can exceed one. Substitute u=t^a, du=a t^(a-1)dt:
integral_0^1 f(t)dt = integral_0^1 b(1-u)^(b-1)du = 1.
This also proves T^a ~ Beta(1,b), used as an independent SciPy test oracle.

The hazard conditions on having survived:

\[
h(t)=\lim_{dt\downarrow0}\frac{P(t<T\le t+dt\mid T>t)}{dt}
=\frac{f(t)}{S(t)}=\frac{ab t^{a-1}}{1-t^a}
=-\frac{d}{dt}\log S(t). \tag{D3}
\]

It is a rate rather than a probability. We evaluate it directly, avoiding f/S
when both probabilities underflow. The shapes a=b=1 give the uniform distribution.

## Two independent competing causes within a block

Suppress block i for readability. Independence gives

\[
P(X>t)=P(T_1>t,T_2>t)=S_1(t)S_2(t)
=(1-t^a)^{b_1+b_2}.\quad X\sim Kw(a,B),\ B=b_1+b_2. \tag{D4}
\]

This is a minimum, not a mixture. It requires a common a across causes.
If the two a values differ, the product survival remains valid, but it is not
generally a Kumaraswamy distribution. The observed failure density is -S_X'(t),
and the observed hazard is h_1(t)+h_2(t).

To observe cause c at t, cause c must fail at t while the other cause survives:

\[
g_c(t)=f_c(t)S_{3-c}(t)
=ab_c t^{a-1}(1-t^a)^{B-1}
=\frac{b_c}{B}f_X(t). \tag{D5}
\]

This joint time/cause **subdensity** integrates to the probability of its cause,
not to one. The two subdensities sum to f_X. Its structure follows independently
from the event definition and agrees with **Reference 2 — Wang (2018)**,
Section 2, Eq. (3), printed p.721 (PDF p.3). **Reference 1 — Wang et al. (2023)**,
Section 2.1, Eqs. (4)-(7), has the analogous block organization for IEED, whose
specific density formulas are not used here.

Integrating (D5) yields

\[
P(\delta=c)=\frac{b_c}{B},\qquad
G_c(t)=P(X\le t,\delta=c)=\frac{b_c}{B}F_X(t). \tag{D6}
\]

These are cause-specific failure probabilities over the whole lifetime and by
mission time, respectively. G_c is cumulative incidence, not F_c. Because the
ratio g_c/f_X is constant, X and delta are independent under this shared-a
model. This is a derived property, not a general fact about competing risks.
The cause-specific observed hazard g_c/S_X equals the latent hazard h_c under
the assumed independent latent model.

The median solves S_X(t)=1/2, giving

\[
\operatorname{MTF}=[1-2^{-1/B}]^{1/a}. \tag{D7}
\]

This uses the Kumaraswamy survival; **Reference 1 — Wang et al. (2023)** uses
an IEED median formula for a different distribution.
Substitution verifies F_X(MTF)=S_X(MTF)=1/2.

## Numerical conventions and limitations

- `Kumaraswamy` exposes pdf/logpdf, cdf, sf/logsf, hazard, and median.
  `CompetingRisks.minimum` exposes the corresponding observed-lifetime functions;
  `cause_pdf`, `cause_logpdf`, `cause_cdf`, and `cause_probability` distinguish the
  joint failures from the latent lifetimes.
- Parameters must be positive finite scalars, with finite b1+b2. Times accept
  NumPy array shapes or scalars; NaNs raise ValueError. Parameter arrays are not
  supported. One object represents one block.
- For t<=0, CDF=0 and SF=1; for t>=1, CDF=1 and SF=0. Density is assigned zero
  at and outside the endpoints. A boundary density value is arbitrary for a
  continuous distribution; this convention does not claim its limiting value
  is zero (it can diverge). There are no point masses at 0 or 1.
- Hazard rejects times outside the open interval; endpoint limits may be infinite.
  Later observed failure data must lie strictly inside the support.
- Compute z=a log(t) and log(1-exp(z)) using `log1p(-exp(z))` when z<=-log(2)
  and `log(-expm1(z))` otherwise. Compute F=-expm1(log S).
  These avoid cancellation near both ends. Logarithms retain information when
  ordinary probabilities underflow. No epsilon replaces a time or parameter.
- Float64 still has finite range: extreme densities/hazards can overflow, tiny
  probabilities can underflow, and extreme medians can round to endpoints.
  Tests cover representative tails, not arbitrary precision or every positive
  parameter value. Boundary MLE behavior is a future estimation concern.

## Validation scope

`tests/test_model.py` checks integration including integrable singularities,
the transformed Beta CDF and density, derivatives F'=f and -(log S)'=h,
survival complement and monotonicity, latent-product versus observed-minimum
identities, subdensity integrals and sums, cause cumulative incidence, median
inversion, uniform special case, support conventions, invalid inputs, scalar
and array behavior, and cancellation/underflow examples.

These checks validate the distribution layer. No likelihood estimates, published
numerical tables, censoring simulator, or MLE uncertainty results have been
reproduced yet. Those require the later milestones and matching assumptions.

## Stage 3: notation and observation convention

Fix the completed schemes R_i and sizes n_i,m_i before testing. For each block
i=1,...,k the data are the ordered times and causes
0<x_i1<...<x_i,m_i<1, delta_ij in {1,2}, with known withdrawals r_ij.
The two independent latent lifetimes follow Kw(a,b_i1), Kw(a,b_i2).
One a>0 is shared across every block and cause; all b_ic>0 are block-specific.
Units are independent within and across blocks. Let

\[
B_i=b_{i1}+b_{i2},\quad
q_{ij}(a)=1-x_{ij}^{a},\quad
d_{ic}=\sum_{j=1}^{m_i}\mathbf1\{\delta_{ij}=c\},\quad
M=\sum_{i=1}^k m_i.
\]

The schedule is treated as fixed, not assigned a probability model. Removal is
uniform random among survivors, independent of latent lifetimes and causes.
The recorded data omit individual unit identities and keep each failure's cause
alongside its ordered time. Thus the sampling density is with respect to
Lebesgue measure on each block's ordered time region and counting measure on
its cause sequence. The likelihood as a function of parameters is that density
evaluated at the observed data; it is not the probability of exact continuous
times and need not be bounded by one.

The original experiment in **Reference 1 — Wang et al. (2023)**, Section 2.1,
Eq. (4), printed p.2741 / PDF p.6, has this block organization but uses IEED.
Its Section 2.2, Eqs. (8)-(9), printed p.2746 / PDF p.11, motivates multiplication
within and across facilities. The following derivation uses our Kumaraswamy
event probabilities, not IEED density or survival expressions.

## Stage 3: failure and withdrawal contributions from first principles

If the unit fails from cause 1 near x, its first latent lifetime must end near x
and its second latent lifetime must exceed x. Independence therefore gives

\[
g_{i1}(x)=f_{i1}(x)S_{i2}(x)
=a b_{i1}x^{a-1}(1-x^a)^{B_i-1}.
\]

Interchanging causes gives

\[
g_{i2}(x)=f_{i2}(x)S_{i1}(x)
=a b_{i2}x^{a-1}(1-x^a)^{B_i-1}. \tag{L1}
\]

Here g_ic(x)dx approximates the joint probability of failure in (x,x+dx) from
cause c. It is not f_ic(x), because that latent density alone does not require
the other cause to survive. It is also not a density conditional on a chosen
cause: g_ic integrates to b_ic/B_i. These statements follow from (D4)-(D6).

A unit withdrawn alive at x supplies the event X>x, with probability
S_i(x)=S_i1(x)S_i2(x)=(1-x^a)^B_i. It has not failed at x and its eventual cause
is not observed. Independence across units gives S_i(x)^r for r survivors.
Combining one failure and its withdrawals gives the parameter-dependent row factor

\[
g_{i,\delta_{ij}}(x_{ij})S_i(x_{ij})^{r_{ij}}
=a b_{i,\delta_{ij}}x_{ij}^{a-1}
q_{ij}(a)^{(r_{ij}+1)B_i-1}. \tag{L2}
\]

The r+1 appears because both the failed unit and the r withdrawn units have
survival factors; the failure density removes one power of 1-x^a. This is an
algebraic statement about distinct units, not r+1 observed failures.

## Stage 3: derive the censoring constant using conditional risk sets

The row factors above do not by themselves establish the joint density of
ordered failure times. Successive failure times are not independent. To derive
the exact constant without counting the same survival event twice, use the
conditional experiment. Write x_i0=0 and

\[
Y_{ij}=n_i-(j-1)-\sum_{\ell<j}r_{i\ell},\qquad
Y_{i,j+1}=Y_{ij}-1-r_{ij}.
\]

Given the history through x_i,j-1, the Y_ij remaining units each have lifetime
conditional on exceeding x_i,j-1. For the next observed failure at x_ij from
cause delta_ij:

- Any one of the Y_ij exchangeable units can fail, giving factor Y_ij.
- That unit contributes g_i,delta(x_ij)/S_i(x_i,j-1).
- The other Y_ij-1 units must survive to x_ij, giving
  [S_i(x_ij)/S_i(x_i,j-1)]^(Y_ij-1).

The conditional next-failure density is therefore

\[
Y_{ij}g_{i,\delta_{ij}}(x_{ij})
\frac{S_i(x_{ij})^{Y_{ij}-1}}{S_i(x_{i,j-1})^{Y_{ij}}}. \tag{L3}
\]

After failure, each eligible set of r_ij survivors has equal selection probability
1/binom(Y_ij-1,r_ij). Because withdrawal identities are not part of these records,
summing over those sets cancels this selection probability. There is no extra
binomial or withdrawal-factorial factor to insert into (L3).

Multiply (L3) over j. At each nonterminal time x_ij, the net survival exponent is
(Y_ij-1)-Y_i,j+1=r_ij. At the final failure, all remaining survivors are removed,
so Y_i,m_i-1=r_i,m_i; no denominator from a further failure exists. The initial
denominator is S_i(0)^n_i=1. Thus

\[
L_i(a,b_{i1},b_{i2};D_i,R_i)
=C_i\prod_{j=1}^{m_i}
g_{i,\delta_{ij}}(x_{ij})S_i(x_{ij})^{r_{ij}},\qquad
C_i=\prod_{j=1}^{m_i}Y_{ij}. \tag{L4}
\]

This is the exact density for our ordered, unlabeled records, conditional on the
scheme. For labeled individual histories an additional selection/labeling
convention can change parameter-free factors; that is a different recorded
sample space. No such identities are stored by our Stage 2 data structures.

As checks: complete data give C_i=n_i!; ordinary Type-II censoring gives
C_i=n_i!/(n_i-m_i)!; one observed failure gives C_i=n_i. For our general
progressive schedule it is the product of risk sets, not generally n_i! or m_i!.

The constants match **Reference 2 — Wang (2018)**, Eq. (3), printed p.721 /
PDF p.3: its c=n product_{j=1}^{m-1}[n-sum_{ell<=j}(r_ell+1)] is precisely
product_j Y_j. Its Eq. (1), printed p.720 / PDF p.2, supplies the underlying
Kumaraswamy density and survival. Restricting its two alpha values to our shared
a yields (L2)-(L4) for one block. We then retain separate b values across blocks.

## Stage 3: block likelihood and combined experiment

Collecting the factors in (L2) by cause gives

\[
L_i=C_i a^{m_i}b_{i1}^{d_{i1}}b_{i2}^{d_{i2}}
\prod_{j=1}^{m_i}x_{ij}^{a-1}
q_{ij}(a)^{(r_{ij}+1)B_i-1}. \tag{L5}
\]

There is no factor m_i!/(d_i1!d_i2!) here: each observed cause is already paired
with its ordered failure time. The powers b_ic^d_ic collect those particular
marks without summing over different cause sequences. Summing (L4) over every
cause sequence instead recovers the time-only progressively censored density
C_i product_j f_Xi(x_ij) S_i(x_ij)^r_ij.

Independence of blocks conditional on the fixed parameters gives

\[
L(a,\{b_{ic}\};D,R)=\prod_{i=1}^k L_i
=C a^M\prod_{i=1}^k b_{i1}^{d_{i1}}b_{i2}^{d_{i2}}
\prod_{i=1}^k\prod_{j=1}^{m_i}
x_{ij}^{a-1}q_{ij}(a)^{(r_{ij}+1)B_i-1},
\quad C=\prod_i C_i. \tag{L6}
\]

Sharing a does not introduce dependence between blocks at fixed parameters.
If a future Bayesian model integrates out a shared random parameter, the
marginal dependence question is different; (L6) remains the conditional
likelihood used with its prior.

## Stage 3: full log-likelihood and constants

Taking logarithms of (L6) gives

\[
\ell=\log C+M\log a+
\sum_i\sum_{c=1}^2 d_{ic}\log b_{ic}
+(a-1)\sum_{i,j}\log x_{ij}
+\sum_{i,j}[(r_{ij}+1)B_i-1]\log(1-x_{ij}^a). \tag{L7}
\]

All logarithms are natural. The full expression retains all factors for our data
convention. Its parameter-free censoring term is log C=sum_ij log Y_ij. This
depends only on the fixed scheme, not on times, causes, or lifetime parameters.

For future algebra define

\[
T_i=\sum_j\log x_{ij},\quad
H_i(a)=\sum_j\log(1-x_{ij}^a),\quad
W_i(a)=-\sum_j(r_{ij}+1)\log(1-x_{ij}^a)>0.
\]

Then an equivalent expression is

\[
\ell=\log C+M\log a+\sum_{i,c}d_{ic}\log b_{ic}
+(a-1)\sum_i T_i-\sum_i B_i W_i(a)-\sum_i H_i(a). \tag{L8}
\]

Only T_i is independent of a among these three summaries. In particular,
-sum_i H_i(a) cannot be dropped as a constant: it depends on the unknown a.
If (a-1)sum_i T_i is expanded as a sum_i T_i - sum_i T_i, the last term is a
time-only, parameter-free constant. Our implementation keeps it even when
`include_constant=False`: that option removes **only log C**, not time factors.

For fixed data and the same model, omitting log C leaves likelihood ratios in
parameters and eventual maximizers unchanged. A posterior normalized over those
same parameters is likewise unchanged because C cancels. However, the full
sampling density, absolute log-likelihood, and any marginal likelihood for that
recorded experiment require the appropriate constant. Comparing different data
conventions or experiments requires matching the underlying sample spaces;
different constants cannot simply be treated as universally interchangeable.

## Stage 3: reference comparison and safeguards

- **Reference 1 — Wang et al. (2023)** supplies the BPCS structure, independent
  block product, and facility-specific cause parameters. Its Eq. (8) displays
  f_2(x)[1-S_1(x)] in one term, an apparent notation error already visually
  verified and recorded in RESEARCH_NOTES.md. Event reasoning requires f_2 S_1,
  and the withdrawal survival must be evaluated at x_ij. We do not reproduce
  those notational errors or transfer its IEED-specific formulas.
- **Reference 2 — Wang (2018)**, Eqs. (1) and (3), supplies matching distribution
  and single-block event structure. Its Eq. (5) places the time-only term
  -sum log x in its additive constant. Equation (L7) retains it within
  (a-1)sum log x. These are equivalent conventions after alpha_1=alpha_2=a.
- No missing causes are introduced. Our Stage 2 `BlockData` rejects them. There
  is no hybrid terminal time, missingness probability, prior, or order restriction.
  No claims about existence, uniqueness, concavity, or asymptotic distributions
  of block-model MLEs are inferred from the reference papers.

## Stage 3: implementation contract and numerical behavior

`../src/likelihood.py` exposes:

| Function | Input and result |
| --- | --- |
| `block_loglikelihood(block,a,b1,b2,include_constant=True)` | Validated `BlockData`; one block's full log-likelihood |
| `block_likelihood(...)` | Exponential of the same block log-likelihood |
| `loglikelihood(data,a,b_by_block,include_constant=True)` | Validated `BPCSExperiment`; one shared a and a mapping from exact block IDs to (b1,b2); sum of block log-likelihoods |
| `likelihood(...)` | Exponential of that combined log-likelihood |
| `log_censoring_constant(data)` | Log C for one block or a whole experiment, using Stage 2 risk sets |

`include_constant` is keyword-only. The parameter mapping must have exactly the
data block IDs; mapping insertion order is irrelevant. Scalars must be real,
positive, and finite; each b1+b2 must be finite. Booleans, strings, complex
parameters, and parameter arrays are rejected. NumPy real scalar values work
with future optimizer/sampler adapters. Block-specific shapes a are deliberately
not accepted by the joint API.

The implementation sums Stage 1 `cause_logpdf` and r times minimum `logsf`.
It therefore reuses the established log1p/expm1 computation of log(1-t^a) and
does not multiply tiny probabilities. `math.fsum` accumulates log terms and
sum log Y computes log C without forming a potentially overflowing product.
No Stage 1 or Stage 2 implementation or test was modified for this milestone.

Invalid input raises ValueError. If mathematically finite logs cannot be
represented in float64, the evaluator raises FloatingPointError instead of
silently producing NaN. Finite-range limitations still apply to extreme shapes.
Ordinary likelihoods can underflow to 0 or overflow to infinity even when their
logarithms are useful; they are not clipped or replaced with epsilon. Use the
log APIs for inference. A future optimizer or sampler must distinguish invalid
parameter proposals from numerical failures; no proposal-handling policy,
prior evaluation, parameter transform, or estimation wrapper is added here.

## Stage 3: work through the two-block teaching data

Use `../experiments/stage3_example.py` with the unchanged Stage 2 teaching data.
The chosen parameters are a=2, (b_A1,b_A2)=(1,2), (b_B1,b_B2)=(.8,1.2).
They are illustrative values, not estimates.

Block A has times (.15,.40,.70), causes (1,2,1), R=(1,0,2), and B_A=3.
The q values are (.9775,.84,.51). Each row's exponent is (r+1)B-1, hence
(5,2,8). The cause counts are (2,1), so (L5) gives

\[
L_A/C_A=2^3(1)^2(2)^1(.15\cdot.40\cdot.70)
(.9775)^5(.84)^2(.51)^8.
\]

The first row, for instance, is one cause-1 failure with factor
2(1)(.15)(.9775)^2, multiplied by the withdrawn survivor's survival
(.9775)^3. Their product is .3(.9775)^5. The terminal row contains one cause-1
failure and two withdrawn survivors, producing power 2+2*3=8. The risk sets
(6,4,3) give C_A=72.

Block B has times (.10,.25,.55,.80), causes (2,1,2,2), R=(0,1,0,2), B_B=2,
and q=(.99,.9375,.6975,.36). The exponents are (1,3,1,5) and counts (1,3):

\[
L_B/C_B=2^4(.8)^1(1.2)^3(.10\cdot.25\cdot.55\cdot.80)
(.99)^1(.9375)^3(.6975)^1(.36)^5.
\]

The risk sets (7,6,4,3) give C_B=504, so C=72*504=36288.
Executing the example gives the following values, rounded for presentation:

| Quantity | A | B | Combined |
| --- | ---: | ---: | ---: |
| Log likelihood omitting C only | -6.246745074472 | -7.085625067089 | -13.332370141561 |
| Log censoring constant | 4.276666119016 | 6.222576268071 | 10.499242387087 |
| Full log-likelihood | -1.970078955456 | -.863048799018 | -2.833127754473 |
| Full likelihood | .139445845770 | .421873910863 | .058828564309 |

For example, T_A=-3.170085660699, W_A(2)=2.239901021181,
H_A(2)=-.870454927531; substitution in (L8) reproduces A's result.
The script prints every failure log-density and every withdrawal log-survival
contribution, so the row calculations can be inspected individually.

## Stage 3: validation and Stage 4 boundary

`../tests/test_likelihood.py` compares against a hand-calculated uniform-minimum
case, the expanded (L7), products of latent densities and survivals, and the
conditional risk-set construction (L3). It tests block factorization, exact
constants, mapping order, summation over all cause sequences, integration of
full sampling densities over ordered times and causes, first-failure cause
probabilities, single-cause blocks, cause relabeling, invalid parameters and
mapping shapes, and probability underflow/density overflow.

These are likelihood correctness checks, not fitted results or reproduction of
the papers' MLE tables. Stage 4 must derive the scores for the shared a and all
b_ic, investigate conditional analytic estimates/profile likelihood, treat
zero-cause-count boundaries, choose parameter transformations and starting
values, check convergence, and compare profile with direct optimization.
At the end of Stage 3, no score solver, MLE, profile optimizer, or uncertainty
method had been implemented. The following section implements Stage 4 only.

## Stage 4: scores for our actual shared-a block model

Start from our full log-likelihood (L8); no IEED derivative is substituted.
For block i define u_ij=-log x_ij>0, w_ij=r_ij+1, and

\[
P_{ij}(a)=\frac{d}{da}\log(1-x_{ij}^a)
=\frac{-x_{ij}^a\log x_{ij}}{1-x_{ij}^a}
=\frac{u_{ij}}{e^{a u_{ij}}-1}>0.
\]

Thus W_i'(a)=-sum_j w_ij P_ij(a)<0 and H_i'(a)=sum_j P_ij(a)>0.
Differentiation of (L8) gives all the interior score equations:

\[
U_{b_{ic}}=\frac{\partial\ell}{\partial b_{ic}}
=\frac{d_{ic}}{b_{ic}}-W_i(a)=0,
\qquad i=1,\ldots,k,\ c=1,2, \tag{M1}
\]

\[
U_a=\frac{\partial\ell}{\partial a}
=\frac{M}{a}+T-\sum_i B_i W_i'(a)-\sum_i H_i'(a)
=\frac{M}{a}+T+\sum_{i,j}(B_iw_{ij}-1)P_{ij}(a)=0,
\quad T=\sum_i T_i. \tag{M2}
\]

There is one equation for the shared a, not a separate a equation for each
facility. All blocks contribute to that equation. The parameter-free censoring
constant has zero derivative, but is retained when reporting log-likelihoods.

The b score balances the observed failures d_ic against the weighted survival
information W_i. The a score balances the time-shape contribution M/a+T against
the changing survival factors; it depends on every failure and withdrawal.

## Stage 4: conditional estimates and the meaning of a zero count

For fixed a, each cause parameter occurs through
d_ic log b_ic - b_ic W_i(a). If d_ic>0, its second derivative is
-d_ic/b_ic^2<0, it tends to -infinity at zero and infinity, and its unique
conditional maximum is

\[
\widehat b_{ic}(a)=\frac{d_{ic}}{W_i(a)}. \tag{M3}
\]

If d_ic=0, the contribution instead equals -b_ic W_i(a), strictly decreasing
for b_ic>0. Its supremum occurs as b_ic decreases to zero. It has **no maximum
in the original positive parameter space**. We report the limiting value zero
with status `boundary_supremum`, not a tiny positive interior estimate.
The one-sided raw derivative at this closure boundary is -W_i(a)<0.
This survival penalty shows that a zero observed count does not mean the
likelihood is independent of that parameter.

Under the convention 0 log 0=0 for this limit, (M3) still gives the closure
estimates. Because d_i1+d_i2=m_i>0,

\[
\widehat B_i(a)=\frac{m_i}{W_i(a)},\qquad
\widehat p_{ic}=\frac{\widehat b_{ic}}{\widehat B_i}
=\frac{d_{ic}}{m_i}. \tag{M4}
\]

One can see the separation by writing b_i1=B_i p_i and b_i2=B_i(1-p_i).
The cause terms become m_i log B_i+d_i1 log p_i+d_i2 log(1-p_i), while
the time/survival terms depend on B_i and a. Observed cause proportions determine
p_i, including p_i=0 or 1 as a boundary limit.
The profile estimate of a depends on times and withdrawals, not on allocation
of the m_i observed failures between causes. This property relies on the
shared-a model and known, noninformatively observed causes.

## Stage 4: profile log-likelihood and profile score

Substitute (M3) into (L8), observing that
sum_c d_ic log(d_ic/W_i)=sum_c d_ic log d_ic-m_i log W_i and
widehat B_i W_i=m_i. The full profile log-likelihood (or boundary supremum) is

\[
\ell_p(a)=\log C+M\log a+(a-1)T
+\sum_{i,c}d_{ic}\log d_{ic}
-\sum_i m_i\log W_i(a)-M-\sum_i H_i(a). \tag{M5}
\]

Every coefficient m_i in front of log W_i is required by the substitution.
The profile derivative is

\[
U_p(a)=\ell_p'(a)
=\frac{M}{a}+T-\sum_i m_i\frac{W_i'(a)}{W_i(a)}-\sum_i H_i'(a)
=U_a(a,\widehat b(a)). \tag{M6}
\]

The last equality also follows by the chain rule: interior conditional b scores
are zero, and absent-cause closure coordinates are fixed at zero for every a.
There is only one nonlinear parameter left to optimize.

**Reference 1 — Wang et al. (2023)**, Section 3.1, Theorem 1 and the profile
construction preceding Theorem 2 (printed pp.2748-2749 / PDF pp.13-14), motivates
block-specific conditional estimation with a common remaining parameter.
Its likelihood and derivatives concern IEED. Equations (M1)-(M6) above are
derived from our Kumaraswamy block likelihood instead.

**Reference 2 — Wang (2018)**, Section 3.2, Theorems 3-4 (printed p.722 /
PDF p.4) and Appendices C-D (printed pp.734-735 / PDF pp.16-17), concerns
the common-alpha single-block Kumaraswamy case. Formula (M3) reduces to its
conditional beta estimate for k=1 with d_ic>0. Our shared-a score sums the
separate block contributions; no general four-parameter or IEED formula was
copied without specialization and derivation. Zero-count and degenerate-block
cases are analyzed explicitly below rather than assuming interior existence.

## Stage 4: existence, uniqueness, and the exceptional unbounded case

We establish these properties for the exact, strictly ordered data contract in
Stage 2. This is a mathematical result for this provisional model, not a claim
that a numerical optimizer always succeeds.

First, the profile is strictly concave in **a**, although this does not imply
concavity in eta=log a. To see this, write

\[
W_i(a)=\sum_jw_{ij}\left[-\log(1-e^{-a u_{ij}})\right]
=\sum_j\sum_{h=1}^\infty\frac{w_{ij}}{h}e^{-a h u_{ij}}.
\]

This is a positive mixture of exponentials in a. The second derivative of
log W_i is the variance of the rates h u_ij under the corresponding normalized
positive weights, hence (log W_i)''>=0. Termwise differentiation is justified:
on any compact interval a in [epsilon,L] with epsilon>0, the finite number of
positive u_ij gives a positive minimum u. The series and its first two
derivatives are uniformly dominated by convergent sums of powers of h times
exp(-epsilon*h*u). The normalized weights therefore have finite first and
second rate moments, so the variance identity applies. Differentiating (M5) twice gives

\[
\ell_p''(a)=
\sum_{i,j}\left[-\frac{1}{a^2}
+\frac{u_{ij}^2e^{-a u_{ij}}}{(1-e^{-a u_{ij}})^2}\right]
-\sum_i m_i(\log W_i(a))''<0. \tag{M7}
\]

Each bracket is strictly negative: with z=a u_ij>0,
z^2 e^{-z}/(1-e^{-z})^2=[z/(2 sinh(z/2))]^2<1. The block sum therefore
extends strict concavity directly to our actual model, without relying on an
IEED uniqueness theorem.

As a decreases to zero, W_i(a)=n_i[-log a]+O(1), since sum_j w_ij=n_i,
and H_i(a)=m_i log a+sum_j log u_ij+O(a). In (M5) the M log a terms cancel,
leaving ell_p(a)=-M log(-log a)+O(1), which tends to -infinity.

As a grows, the largest observed time x_i,m_i dominates W_i:
W_i(a) is asymptotic to w_i,m_i exp(-a u_i,m_i). Consequently,

\[
\ell_p(a)=M\log a-aD+O(1),\qquad
D=\sum_i\sum_j\log\!\left(\frac{x_{i,m_i}}{x_{ij}}\right)\ge0. \tag{M8}
\]

If at least one block has m_i>=2, strict within-block ordering gives D>0.
The profile tends to -infinity at both ends and is continuous and strictly
concave, so it has a unique finite maximizer with U_p=0. If every d_ic>0,
the conditional estimates are positive and the full model has a unique interior
MLE. If any d_ic=0, the same finite profile maximum gives the unique maximum
on the relevant closure face, but only a supremum in the original open domain.

If every block has m_i=1, then D=0 and ell_p(a) grows as M log a plus a constant.
The likelihood is unbounded above; there is no finite maximizing vector or
finite boundary supremum. Both fitting methods report `unbounded_likelihood`,
with a=None, no finite b vector, and log-likelihood=+infinity. They do not return
the arbitrary endpoint of a numerical search. A block with m_i=1 is otherwise
allowed when another block supplies at least two distinct failures.

The closure b_ic=0 is a limit, not a valid input to the original continuous
Kw(a,b_ic) class. Its latent law concentrates at the upper endpoint as b_ic
decreases to zero. Never pass a boundary result directly to the strictly
positive Stage 1/3 interfaces without explicitly using the limiting model.

## Stage 4: numerical implementation and diagnostics

`../src/mle.py` exposes `score`, `conditional_estimates`, `profile_score`,
`profile_loglikelihood`, `fit_mle` (profile), and `fit_direct` (joint).
The existing Stage 1-3 source and test files are unchanged.

The profile implementation evaluates (M5) with stable log W_i and the convention
0 log 0=0. It solves a U_p(a)=0 in eta=log a using Brent's bracketed method.
Starting at initial_a (default 1), it doubles or halves a until it finds the
opposite score sign. The defaults allow 60 expansions and 200 root iterations;
the log-coordinate absolute/relative root tolerances are 1e-12.
Monotonicity of U_p in a and (M7)-(M8) justify the unique root. The sign of
a U_p is the same as U_p because a>0.

For stability, log q comes from the existing Kumaraswamy log-survival function
with b=1. Let v=-log q. Compute log W with log-sum-exp of log w+log v.
For a*u>36, log v=-a*u to float64 accuracy (the omitted relative correction is
less than 1.2e-16). Weighted ratios compute -W'/W without dividing two
underflowed numbers. For z=a*u near zero,
1-z/(exp(z)-1)=z/2-z^2/12+z^4/720+O(z^6), avoiding cancellation in the score.
Conditional b values are exponentiated only after log d-log W has been formed.

For an interior result, the reported objective is evaluated by the unchanged
Stage 3 `loglikelihood`. The profile expression is checked against it. For an
absent cause, exact evaluation on the closure face uses a linear identity:
replace b_ic=0 temporarily by t=1/(1+W_i), evaluate Stage 3, then add t*W_i.
Because d_ic=0, ell(b_ic=0)=ell(b_ic=t)+t*W_i exactly. This uses a positive
reference for evaluation, not an epsilon estimate or an approximate boundary.

The independent direct optimizer does not use the profile fit or conditional
estimates to initialize or update parameters. It jointly optimizes eta=log a
and gamma_ic=log b_ic for positive-count causes using L-BFGS-B and the Stage 3
objective (with the same exact closure correction where needed). Its gradient is

\[
\frac{\partial\ell}{\partial\eta}=aU_a,\qquad
\frac{\partial\ell}{\partial\gamma_{ic}}=d_{ic}-b_{ic}W_i(a).
\]

Absent causes are fixed at zero by (M1), not optimized on a log scale.
Default starting a values are (.5,1,2,5), each with positive b=1. Custom starts
use an (a,block-ID-to-(b1,b2)) pair and must lie in the original positive domain;
values for absent causes are then discarded in favor of the exact zero limit.
No profile estimate is used as a warm start, so the joint method provides an
independent optimization check rather than merely re-evaluating the profile fit.

The direct method has log-coordinate guards [-700,700] to prevent float64
exponentiation overflow. These are numerical safeguards, not scientific
parameter restrictions. Reaching a guard cannot be reported as a converged MLE.
Numerically invalid trial points are rejected; their count is recorded.

The returned `MLEResult` contains a, `b_by_block`, full log-likelihood,
`converged`, `status`, method, message, and diagnostics. Success requires the
solver's success flag, an absolute active log-parameter score norm no larger
than score_tol*M (default score_tol=1e-7), and agreement between the conditional
profile bound and joint likelihood at the candidate. Direct fits must also be
away from their numerical guards. Boundary one-sided scores -W_i are recorded
separately, since those coordinates do not have interior zero scores.

| Status | Interpretation |
| --- | --- |
| `interior_mle` with converged=True | Numerically verified finite positive MLE |
| `boundary_supremum` with converged=True | Finite closure optimum; no MLE in the strictly positive domain |
| `unbounded_likelihood` | No finite maximum; all blocks have one observed failure |
| `optimization_failed` | Iteration/bracketing budget, gradient, guard, or objective check failed; any candidate is not a certified MLE |
| `numerical_failure` | Float64 arithmetic cannot represent a required calculation or conditional estimate |

Bracketing exhaustion is not proof of a statistical boundary. In particular,
extremely close or very small failure times can require unrepresentably large
conditional b values. The implementation reports failure rather than clamping
the estimate. The L-BFGS internal curvature approximation is used only for
optimization; no Hessian, standard error, information matrix, or interval is
returned as an uncertainty estimate.

## Stage 4: fitted two-block teaching example

Run `.venv/bin/python -m experiments.stage4_example` from the repository root.
The data remain the Stage 2 hand-constructed rows; parameters are now estimated,
unlike the chosen-parameter evaluations in Stage 3.

| Parameter/result | Profile | Independent direct |
| --- | ---: | ---: |
| a | 1.173959840838 | 1.173959842586 |
| b_A1 | .517708109960 | .517708110913 |
| b_A2 | .258854054980 | .258854056901 |
| b_B1 | .178753429726 | .178753430518 |
| b_B2 | .536260289178 | .536260288806 |
| Full log-likelihood | 1.070678947337 | 1.070678947337 |
| Maximum absolute active log-parameter score | 4.44e-16 | 6.07e-9 |

Both methods report `interior_mle` and converged=True. Profile root finding
takes 8 iterations after one bracket expansion from a=1 to a=2. The direct
method uses four independent starts. The full log-likelihoods differ by about
8.9e-16 in this environment. At the profile fit, all five raw scores are zero
to displayed floating-point accuracy (largest cause score about 8.9e-16).

At the fitted a, the conditional cause ratios are exactly d_A1/m_A=2/3 and
d_B1/m_B=1/4. This explains the fitted eventual cause-1 probabilities without
confusing them with cumulative incidence by a specified time. The fitted
log-likelihood exceeds the Stage 3 chosen-parameter value -2.833127754473.
Positive log-likelihood is allowed because these are continuous-time densities.
These are teaching-data fits, not published numerical reproductions or evidence
of simulation performance.

## Stage 4: tests and limits of parameter-recovery checks

`../tests/test_mle.py` compares the raw scores to finite differences of the
unchanged Stage 3 likelihood, conditional estimates to independent inner
numerical maximization, and the profile score to finite differences and the
envelope identity. It checks the profile/direct fits, remote starting values,
arbitrary block counts, cause proportions, finite boundary limits and one-sided
scores, the unbounded single-failure case, exhausted solver budgets, numerical
range failures, and invalid inputs.

Known-parameter recovery uses deterministic distribution-quantile fixtures,
not random simulations. For each block, p_j=(j-.5)/n and
x_j=[1-(1-p_j)^(1/B_i)]^(1/a_true), with completed schemes R=0. Blocks of sizes
600 and 500 use B_A=3 and B_B=2, with deterministic cause proportions 1/3 and
2/5. Three true a values (.6,2,5) give fitted a values approximately
(.600904768,2.003015892,5.007539731); fitted b_A=(1.002465925,2.004931851)
and b_B=(.801763145,1.202644718). The finite grid approximates its population,
so exact recovery is not expected. This deterministic test is not a Monte Carlo
study, does not generate the BPCS removal process, and establishes no empirical
bias, consistency, coverage, or uncertainty result.

Stage 5 remains unstarted. Observed Fisher information, standard errors,
confidence intervals, profile intervals, and published numerical uncertainty
validation will require a separate milestone. Bayesian methods and simulations
also remain deferred.

Validation on 2026-10-23: the Stage 4 suite passed 39 cases, and the combined
suite reported **170 passed in 1.11 seconds**. The executed teaching example
produced the estimates above. Compilation and Git whitespace checks also passed.

## Order-restricted MLE: scope, interpretation, and literature

This milestone adds only the user-requested provisional within-block restriction
b_i1 <= b_i2, independently for every i. All a,b_ic remain strictly positive;
non-strict equality is permitted. No cross-block order, missing causes, hybrid
stopping, or changed censoring likelihood is introduced. No Jacobian belongs
in this likelihood reparameterization: we are maximizing a sampling density
at the same parameters, not transforming a probability density over parameters.

The restriction has a specific interpretation under the shared-a model:
h_i1(t)<=h_i2(t), P(delta=1)<=1/2, and
S_i1(t)>=S_i2(t) for every interior t, so latent cause-1 lifetime is
stochastically larger. Indeed, h_ic=ab_ic t^(a-1)/(1-t^a), and the base
1-t^a lies in (0,1). This is a comparison of causes WITHIN a block, not of
facilities, observed failure times, or realized failure counts. It is provisional
and must eventually be justified by scientific knowledge about the labeled
causes. Choosing the direction after seeing counts would change its meaning.

Literature provenance:

- **Reference 1 — Wang et al. (2023)**, Section 2.1 and Eq. (4), motivates block organization; its
  distribution is IEED, so its parameter formulas are not used here.
- **Reference 2 — Wang (2018)**, Section 3.2, Theorems 3-4 (PDF p.4 / printed p.722), treats
  common-alpha single-block Kumaraswamy estimation without this order.
- **Reference 3 — Mahto et al. (2022)**, Section 4.1, Theorems 4.1-4.2 (PDF pp.11-12 / printed
  pp.2074-2075), already establishes conditional pooling at lambda_1=lambda_2
  when cause-1 counts dominate, and an unchanged common-alpha estimate under
  lambda_1<=lambda_2 in its single-experiment generalized progressive hybrid
  setting. Its theorem statement assumes positive counts for both causes and
  also permits partially observed causes. Here all causes are known and each
  completed block has its own pair and fixed progressive Type-II schedule.

The calculations below derive these results for our exact multi-block model,
including its zero-count limits and unbounded cases. They are not claims of
novelty or estimator superiority. We do not import the hybrid stopping rule,
missingness model, priors, or uncertainty formulas of
**Reference 3 — Mahto et al. (2022)**.

## Reparameterization and constrained conditional maximization

Starting from (L8), put B_i=b_i1+b_i2 and p_i=b_i1/B_i. The map is one-to-one
between positive pairs and B_i>0, 0<p_i<1, with inverse
b_i1=B_i p_i, b_i2=B_i(1-p_i). The restriction is 0<p_i<=1/2. Thus maximize

\[
\ell(a,\mathbf B,\mathbf p)=\log C+M\log a+(a-1)T-\sum_i H_i(a)
+\sum_i\{m_i\log B_i-B_iW_i(a)+g_i(p_i)\},\quad
 g_i(p)=d_{i1}\log p+d_{i2}\log(1-p), \tag{R1}
\]

subject to a>0, B_i>0, 0<p_i<=1/2, with
W_i(a)=-sum_j(r_ij+1)log(1-x_ij^a)>0 and d_i1+d_i2=m_i>=1.
For fixed a, the B and p problems separate, including across blocks.

For B_i, derivative m_i/B_i-W_i vanishes at

\[
\widehat B_i(a)=\frac{m_i}{W_i(a)}. \tag{R2}
\]

The second derivative -m_i/B_i^2 is strictly negative, and the objective tends
to -infinity at either endpoint of B_i>0, proving a unique conditional maximum.

For positive counts, g_i'(p)=d_i1/p-d_i2/(1-p) and
 g_i''(p)=-d_i1/p^2-d_i2/(1-p)^2<0. Its unconstrained root is d_i1/m_i.
If d_i1<d_i2 it lies below 1/2 and is unchanged. If d_i1=d_i2 the root is
exactly 1/2, feasible and active with zero order multiplier. If d_i1>d_i2,
g_i'(1/2)=2(d_i1-d_i2)>0; strict concavity implies g_i is increasing throughout
the feasible interval, and its unique maximum is the INCLUDED endpoint 1/2.
Consequently, with the zero-count convention qualified below,

\[
\widehat p_i^R=\min\{d_{i1}/m_i,1/2\},\qquad
(\widehat b_{i1}^R(a),\widehat b_{i2}^R(a))=
\begin{cases}
(d_{i1}/W_i(a),d_{i2}/W_i(a)),&d_{i1}\le d_{i2},\\
(m_i/[2W_i(a)],m_i/[2W_i(a)]),&d_{i1}>d_{i2}.
\end{cases} \tag{R3}
\]

The equality-count case agrees in both branches. Pooling preserves the total;
clipping only b_i1, swapping the pair, or sorting the cause labels would not
maximize this labeled-data likelihood.

For a KKT check, extend p's lower endpoint only when its count is zero. For
maximization with p<=1/2, p>=0, stationarity is g_i'(p)-mu_i+nu_i=0, with
nonnegative upper/lower multipliers and complementary slackness. At the upper
endpoint mu_i=2(d_i1-d_i2)>=0 and nu_i=0. At an interior p both are zero.
For d_i1=0,p=0 the finite one-sided derivative is -m_i, so nu_i=m_i,mu_i=0.
This distinguishes an active positive equality from a zero-parameter boundary;
an active equality does not require both original cause scores to vanish.

Equivalently, use the original positive parameters with constraint
b_i1-b_i2<=0. At a strictly pooled block and fixed optimal B_i, the raw scores
are U_b1=(d_i1-d_i2)/B_i and U_b2=-U_b1. The multiplier
lambda_i=(d_i1-d_i2)/B_i>=0 gives U_b1-lambda_i=0,
U_b2+lambda_i=0 and lambda_i(b_i1-b_i2)=0. At equal counts lambda_i=0.
For fixed a the separate B and p objectives are strictly concave on their
respective feasible intervals (or have the stated one-sided boundary limits).
Thus these KKT conditions identify their unique global optima, not merely
stationary candidates. Global uniqueness in a then follows from (R5) and (M7).

## Restricted profile and equality of the shared-a estimates

Substitute (R2)-(R3) into (R1), interpreting 0 log 0 by its limit:

\[
\ell_p^R(a)=\log C+M\log a+(a-1)T-\sum_iH_i(a)
+\sum_i\{m_i\log m_i-m_i\log W_i(a)-m_i+g_i(\widehat p_i^R)\}.
\tag{R4}
\]

The unrestricted profile uses p_i^U=d_i1/m_i instead. Therefore

\[
\ell_p^R(a)=\ell_p^U(a)-\Delta,\qquad
\Delta=\sum_i[g_i(p_i^U)-g_i(\widehat p_i^R)]\ge0. \tag{R5}
\]

Each term is nonnegative because p_i^U maximizes g_i over the larger space.
For violating blocks it equals
 d_i1 log(2d_i1/m_i)+d_i2 log(2d_i2/m_i), with 0 log 0=0;
for nonviolating blocks it is zero. It is strictly positive for every violating
block (including d_i2=0), and depends only on the observed counts, not a.
Hence both profiles have identical derivatives, curvature, tail behavior, and
argmax. Whenever finite,

\[
\widehat a_R=\widehat a_U,\quad
\widehat B_{i,R}=\widehat B_{i,U},\quad
\sup\ell_R=\sup\ell_U-\Delta\le\sup\ell_U. \tag{R6}
\]

Equality of finite suprema holds exactly when every d_i1<=d_i2. For unbounded
profiles both suprema are +infinity; do not subtract infinity from infinity.
Their difference is still the finite constant Delta at any fixed a.
This invariance relies on the within-block proportion restriction only. A
cross-block constraint or constraint involving a need not preserve the profile.
Per-block minimum survival, hazard, and median depend on a and B_i, so their
plug-in values also remain unchanged; cause probabilities generally change.
No uncertainty or testing distribution is inferred from this identity.

## Zero counts, attainment, and unbounded likelihood

Since m_i>0, both counts cannot be zero.

| Observed counts | Restricted p and pair at fixed a | Interpretation |
| --- | --- | --- |
| d_i1=0, d_i2=m_i | p approaches 0; (0,m_i/W_i) in the closure | g=m_i log(1-p) strictly decreases; no maximum for p>0; finite boundary supremum |
| d_i1=m_i, d_i2=0 | p=1/2; both b=m_i/(2W_i)>0 | g=m_i log p increases; attained positive equality, despite unrestricted boundary |
| Both positive, d_i1<d_i2 | p=d_i1/m_i<1/2 | Attained positive pair with inactive order |
| Both positive, d_i1>=d_i2 | p=1/2 | Attained positive pair with active order |

The zero in the first row is not a valid latent Kw(a,0) parameter. It denotes
a limit in the original model, not an epsilon substitute or changed probability
class. To approach it feasibly at fixed optimal B, take b_i1=B*epsilon and
b_i2=B*(1-epsilon), epsilon decreasing to zero. Every such vector is positive
and ordered for epsilon<=1/2; its likelihood tends to the reported supremum.

By (R5), the strict concavity and tail argument (M7)-(M8) applies unchanged:
if any block has at least two strictly ordered failures, there is a unique
finite maximizing a and positive B_i. The restricted maximum is attained in
its original domain precisely when EVERY block has d_i1>0. Blocks with
no cause-2 failures are now allowed for attainment. If any block has d_i1=0,
only a finite closure supremum exists, with b_i1=0 in those blocks.
If every block has m_i=1, the restricted profile is unbounded as a tends to
infinity, because the finite penalty cannot remove M log a growth. Even a
single all-cause-1 block has no finite maximum in that exceptional design.
One single-failure block alongside an identifying multi-failure block is allowed.

## Implementation, diagnostics, and independent validation

`src/order_restricted.py::fit_order_restricted_mle` returns the existing
MLEResult and accepts the same profile controls as fit_mle plus positive finite
constraint_tol (default 1e-10). It reuses the unchanged fit_mle root and stable
_Problem summaries; nonconvergence/unbounded/range failures are propagated.
Already feasible pairs, including equal-count pairs, are retained exactly.
Only strictly violating pairs are recomputed in log scale
as exp(log m_i-log W_i-log 2), with no generic constrained optimizer in production.
No existing distribution, censoring, likelihood, or unrestricted MLE source
module was modified.

For positive pairs the final objective is the original Stage 3 loglikelihood,
including log C. When b_i1=0, temporarily replace it by the positive b_i2 and
add b_i2*W_i back: the absent-cause log-likelihood term is exactly -b_i1*W_i.
This corrects a reference evaluation exactly; it does not approximate a zero by
a small fitted parameter. Positive b_i2 in all-cause-1 equality blocks is retained.

Success checks include finite representable pairs/totals, the normalized order
residual (b_i1-b_i2)/B_i<=constraint_tol, the shared log-a and log-total scores,
and agreement between full joint evaluation and ell_p^U(a)-Delta. The score
threshold is score_tol*M and the absolute objective tolerance is
max(1e-8,score_tol*M), matching the existing profile check. Cause probabilities
are set by (R3), and upper/lower p multipliers are reported. Diagnostics include
active_constraints (all equality blocks), pooled_blocks (strict count violations),
boundary_parameters (actual zero coordinates), fitted probabilities, order
residuals, relative block-total preservation residuals, scores, objective
penalty/check, and nested unrestricted diagnostics.
No raw unconstrained b-score-zero requirement is imposed on an active equality.

`interior_mle` means an attained positive-parameter fit, even if on the ORDER
boundary. `boundary_supremum` means an actual zero b limit. Both use
converged=True only after numerical checks. This retains the existing result
contract; failed candidates must not be interpreted as MLEs. Numerical failures
return no parameter vector. No derivative-based uncertainty is reported.
The compatibility `fit(data, restriction='b_i1 <= b_i2', **options)` accepts only
this restriction and rejects others explicitly.

`tests/test_order_restricted.py` checks inactive, violating, equality-count,
zero-count, mixed blocks, arbitrary k and block order, common-a/total invariance,
likelihood ordering, independent constrained optimization, unbounded designs,
solver budgets, input validation, arithmetic failure, objective inconsistency,
and preservation of unrestricted results. Its independent oracle jointly fits
(log a,b11,b12,...) using SLSQP, nonnegative raw b bounds and linear b2-b1>=0
constraints with finite-difference gradients. It evaluates an independently
expanded row likelihood using xlogy for zero-count limits. It has no profile
updates, analytical conditional estimates, fitted starts, or preset active set.
Two fixed feasible starting vectors and a finite numerical log-a guard are used;
these fixtures' solutions lie well inside that guard. Agreement is numerical
validation of these examples, not a general solver proof or simulation study.

The audit additionally supplies an independently runnable verifier,
`experiments/within_block_validation.py::fit_slsqp`. It jointly optimizes
(eta=log(a/a_scale),beta_i=log B_i,p_i), with independent bounds
0<=p_i<=1/2 and floating-point guards for log a/log B. Here
a_scale=1/mean(-log x), using all observed times only as a coordinate scale;
the data and likelihood are unchanged. Three fixed relative starting shapes,
totals, and feasible proportions are used. There is no analytical conditional
update, profile warm start, or prescribed active set. The independent expanded
objective retains risk-set constants and zero-count limits. Its analytic
gradient is separately verified against finite differences.

Verification requires SLSQP success, finite objective, feasible proportions,
no numerical-guard contact, and projected score norm <=1e-6*M. For maximization,
a lower-bound p=0 requires g'(p)<=0 and an upper-bound p=1/2 requires g'(p)>=0;
the projected residual keeps violations of these conditions rather than
incorrectly requiring a zero raw derivative at an endpoint. A zero-count p
within 1e-10 of zero may be cleaned up to the exact closure endpoint after
optimization; objective and KKT checks are repeated there. No positive floor
is used as a fitted boundary parameter. Each start's convergence, feasibility,
guard status, and residual norm are recorded.

The original raw-b test oracle's fixed log-a guard [-15,15] cannot verify all
extreme valid datasets. For instance, times (1-1e-8,1-5e-9,1-1e-9) give
a_hat approximately 2.00352774e8, exceeding exp(15). This is a limitation of
that finite test search, not of the analytical estimator. The scaled joint
verifier agrees with the analytical estimate in this case, at very small times,
and across the representable endpoints of the support. It is a second focused
validation method; the original oracle and its 52 tests are preserved.

## Reproduce the restricted teaching example

Run `.venv/bin/python -m experiments.order_restricted_example`. Block A has
counts (2,1), so p_A changes from 2/3 to 1/2 and its total .776562164940 is
split equally. B has counts (1,3), so p_B stays 1/4.

| Quantity | Unrestricted | Restricted |
| --- | ---: | ---: |
| a | 1.173959840838 | 1.173959840838 |
| b_A1 | .517708109960 | .388281082470 |
| b_A2 | .258854054980 | .388281082470 |
| b_B1 | .178753429726 | .178753429726 |
| b_B2 | .536260289178 | .536260289178 |
| Full log-likelihood | 1.070678947337 | .900779910542 |
| Status | interior_mle | interior_mle |
| Converged | True | True |

A's equality is active, B's is inactive, and no zero boundaries occur.
Delta=2 log(4/3)+log(2/3)=.169899036795. Tests compare these estimates with
the separate joint constrained optimizer. These remain hand-constructed data;
no published tables, real-data fit, bias/MSE claim, or superiority result follows.

The expanded example prints all seven input rows, unrestricted/restricted and
independent SLSQP fits, cause probabilities, pooling decisions, and minimum
survival at t=.5. S_A(.5)=.634625780839 and S_B(.5)=.657914837980 under both
unrestricted and restricted fits. The independently fitted SLSQP values are
a=1.173959821550, A=(.388281083610,.388281083610),
B=(.178753439640,.536260336209); full log L=.900779910542.
All three SLSQP starts pass their numerical checks. The maximum absolute
parameter difference is about 4.71e-8 and absolute log-likelihood difference
is 1.93e-14 (rounded upward). These differences are within the declared
verification tolerances; no material disagreement remains in these fixtures.

Create the machine-readable report, including input rows and per-start details:

```sh
.venv/bin/python -m experiments.within_block_validation --output results/within_block_validation.json
```

`tests/test_within_block_validation.py` adds focused checks of exact retention
for already-equal pairs, raw KKT scores at positive equalities, survival/total
invariance, the count-only profile penalty over a grid (including both zero-count
directions), finite-difference gradients, nine extreme-time/mark cases, numerical
range failure, solver failure, and the reproducible report. The stress cases
use times as small as nextafter(0,1), as large as nextafter(1,0), very small
positive times, and nearly coincident times with unrepresentable fitted b values.

## Remaining research limits for the order restriction

Priority 1 is scientific justification of the cause labels and direction.
Focused extreme-time, boundary, and near-coincident-time checks have now been
added. They do not exhaust all shapes, block counts, or float64 range limits.
Production depends on
the unrestricted fit succeeding; exceptionally a restricted fit may be
representable even when the unrestricted conditional parameters are not. That
fallback is not implemented. The independent oracle's finite guards and selected
fixtures are not exhaustive verification over all possible data.
Priority 3, only after a separately authorized scope, is to reconsider other
scientifically motivated restrictions; their profiles must be derived afresh.
No Bayesian inference, intervals, experimental simulation study, or real-data
analysis is added in this milestone.

## Classical uncertainty: I1–I7 (2026-10-26)

### I1. Factorization and derivatives

These are derivations for **our provisional common-a, block-specific-b model**.
Reference 1 — Wang et al. (2023), Section 2.1, PDF p.6, supports the fixed
block-wise progressive Type-II experiment and common-mechanism/facility-specific
parameter structure; its IEED derivatives are not used. Reference 2 — Wang
(2018), Section 3.2, Theorems 3–4, equations (8)–(9), supports the single-block
common-shape Kumaraswamy baseline. Its Section 3.3 and Appendix F motivate
observed-information ACI, with the qualifications in I3. Reference 3 — Mahto
et al. (2022) concerns a different censoring design; no hybrid likelihood or
Bayesian interval is imported here.

Write w_ij=1+r_ij, u_ij=-log x_ij>0, z_ij=a u_ij,
L_ij=log(1-exp(-z_ij)), W_i=-sum_j w_ij L_ij,
H_i=sum_j L_ij, T=sum_ij log x_ij, and M=sum_i m_i.
With B_i=b_i1+b_i2 and p_i=b_i1/B_i, substituting b_i1=B_i p_i and
b_i2=B_i(1-p_i) in the verified likelihood gives

\[
\ell=\log C+M\log a+(a-1)T+
\sum_i\{m_i\log B_i-B_iW_i(a)-H_i(a)
+d_{i1}\log p_i+d_{i2}\log(1-p_i)\}. \tag{I1}
\]

This is a reparameterization of a likelihood, so **no density Jacobian** is
introduced. C depends on risk sets only; its derivatives vanish. In plain
language, failure times estimate a and each total B, while the cause labels
estimate the share p. The two parts separate exactly in the likelihood.

Direct differentiation, holding observations fixed, yields

\[
P_{ij}=L'_{ij}=\frac{u_{ij}}{e^{a u_{ij}}-1},\qquad
L''_{ij}=-Q_{ij},\quad
Q_{ij}=\frac{u_{ij}^2e^{-a u_{ij}}}{(1-e^{-a u_{ij}})^2}.
\]

Hence W'_i=-sum wP, W''_i=sum wQ, H'_i=sum P,
and H''_i=-sum Q. The complete first derivatives are

\[
U_a=M/a+T-\sum_i B_iW'_i-\sum_i H'_i,\quad
U_{B_i}=m_i/B_i-W_i,\quad
U_{p_i}=d_{i1}/p_i-d_{i2}/(1-p_i). \tag{I2}
\]

### I2. Analytical observed information

In q=(a,B_1,...,B_k,p_1,...,p_k), all nonzero entries of J=-Hessian(ell) are

\[
\begin{split}
J_{aa}&=M/a^2+\sum_{ij}(B_iw_{ij}-1)Q_{ij},\\
J_{a B_i}&=J_{B_i a}=W'_i=-\sum_jw_{ij}P_{ij},\\
J_{B_iB_i}&=m_i/B_i^2,\\
J_{p_ip_i}&=d_{i1}/p_i^2+d_{i2}/(1-p_i)^2.
\end{split}\tag{I3}
\]

All different-block B/B, different-block p/p, a/p and B/p entries vanish.
For example, differentiating U_a in B_i gives -W'_i, so negating this
second derivative gives W'_i (negative), which fixes the cross-entry sign.
In original coordinates theta=(a,b_11,b_12,...), J_aa is unchanged,
J_(a,b_ic)=W'_i, J_(b_ic,b_ic)=d_ic/b_ic²; all b/b off-diagonals vanish.
These are **observed**, data-dependent matrices. Expected Fisher information
would require averaging over this censoring experiment and is not implemented.

For completeness, away from a stationary point Hessians do not transform like
covariances. Let A=dq/dtheta. Then

\[
J_\theta=A^TJ_qA-\sum_i U_{p_i}\nabla_\theta^2p_i.\tag{I4}
\]

The B mapping is linear and contributes no second derivative. In block i,
p_b1=(1-p)/B, p_b2=-p/B,
p_b1b1=-2(1-p)/B², p_b2b2=2p/B²,
p_b1b2=(2p-1)/B². The correction vanishes at an unrestricted interior MLE,
but generally does **not** vanish at an active constrained fit. Tests check
this identity away from an MLE and reject the naive congruence by example.

Numerical stability: compute L through the existing model's log survival,
and evaluate aP=z exp(-z)/(1-exp(-z)) and
 a²Q=z² exp(-z)/(1-exp(-z))² in logarithms. Near zero use
1-aP=z/2-z²/12+z⁴/720-z⁶/30240 and
1-a²Q=z²/12-z⁴/240+z⁶/6048. Evaluate
J_aa=sum[(1-a²Q)+B_i w_ij a²Q]/a² to avoid cancellation.
Unrepresentable derivatives fail explicitly; exact zero cause parameters
are never replaced by epsilon. Full information requires interior p; the
(a,B) submatrix remains evaluable at a zero-cause closure limit.

### I3. Covariance, standard errors, and approximate intervals

For an unrestricted, finite, identifiable interior MLE under regular
large-sample conditions (fixed k, sufficient failures per block and information
in the time experiment), V_q=J_q(hat q)^(-1) estimates its covariance.
This is not a finite-sample guarantee. At the fitted p=d1/m, J_pp=m/[p(1-p)],
so its inverse is p(1-p)/m and all estimated time/p covariances are zero.
The inverse time matrix retains the common-a coupling between block totals.

To calculate covariance, first require a finite square symmetric matrix with
positive diagonal. Set s_j=sqrt(J_jj), R_jl=J_jl/(s_j s_l), require a Cholesky
factorization and an acceptable condition number of R (default <=10^12).
Solve R X=Identity using the factor and set V_jl=X_jl/(s_j s_l).
Report raw and equilibrated conditioning. Raw conditioning depends strongly
on parameter units. There is no ridge adjustment or pseudoinverse. If the
checks fail, regular SE/ACI are unavailable with an explicit reason; exact
probability sets still remain available. The public matrix function may
return an indefinite matrix away from an MLE; the covariance function rejects it.

For theta=g(q), the delta covariance is V_theta=G V_q G^T, G=dg/dq:
b_i1 derivatives in (B_i,p_i) are (p_i,B_i), while b_i2 derivatives are
(1-p_i,-B_i); the a derivative is identity. Thus, for example,
Var(b_i1)=p_i² Var(B_i)+B_i² Var(p_i) at the unrestricted fit,
and Cov(b_i1,b_i2)=p_i(1-p_i)Var(B_i)-B_i²Var(p_i).
Standard errors are square roots of diagonal variances.

For confidence 1-alpha and z=Phi^(-1)(1-alpha/2), the marginal normal interval
is estimate +/- z SE. Raw normal intervals may leave the support. For a
positive parameter v we also return

\[
[\hat v\exp(-z\,SE/\hat v),\ \hat v\exp(z\,SE/\hat v)].\tag{I5}
\]

This follows from Var(log vhat) approximately Var(vhat)/vhat².
**Reference 2 — Wang (2018), Section 3.3, PDF p.5 prints the delta variance
with denominator hat lambda_j rather than its square, and Appendix F labels
raw negative second derivatives as information entries although Section 3.3
uses the negative Hessian.** We derive the square and sign above, checked
against numerical differentiation. These are paper notation/formula
inconsistencies; no preexisting repository algorithm required correction.
For an interior probability use logit(vhat) +/- z SE/[vhat(1-vhat)], then
apply expit. Intervals are marginal and approximate, not simultaneous.
Floating-point overflow/underflow of transformed endpoints is reported as
unavailable; no arbitrary clipping replaces the interval.

### I4. Restricted and zero-boundary inference

In this model cause and minimum time are independent: g_i1(t)=p_i f_min,i(t).
Fixed progressive withdrawals chosen independently of cause and remaining
latent lifetimes preserve iid Bernoulli(p_i) labels at the m_i failures.
Consequently d_i1 ~ Binomial(m_i,p_i), also conditionally on the observed
failure-time experiment. Adaptive cause-dependent withdrawals, missing causes,
or a random stopping count require a new argument and are outside this API.

For d successes out of m, an equal-tailed Clopper–Pearson interval [L,U] has
L=0 if d=0, otherwise BetaQuantile(alpha/2;d,m-d+1);
U=1 if d=m, otherwise BetaQuantile(1-alpha/2;d+1,m-d).
Under p<=1/2 return the **set intersection** [L,U] intersect [0,1/2].
If L>1/2, the result is empty, represented with status `empty` and None
endpoints. Never replace it by a singleton at 1/2. For every true allowed p,
{p lies in the intersected set} equals {p lies in the original set}; thus
marginal coverage remains at least 1-alpha under the binomial model.
This does not imply coverage under a false order, simultaneous coverage over
blocks, or that the confidence set always contains the constrained estimate.
It can be conservative and wide. Empty sets indicate tension with the order
at this confidence level; this implementation adds no separate testing rule.
For cause 2 complement endpoints, preserving the empty-set status.
This exact set construction is our implementation choice derived from I1;
we do not attribute it to a reference paper's restricted-interval procedure.

The estimator ptilde=min(d/m,1/2) has an active equality when d1>=d2,
is inactive for 0<d1<d2, and lies at the zero boundary if d1=0.
Equality with positive b is distinct from a zero parameter. We use exact
sets uniformly for restricted p and do not return ordinary restricted p SEs.
At all-cause-1 samples the restricted estimate is positive equality; at
all-cause-2 samples b1=0 is a closure supremum. Neither implies a regular
full-parameter covariance. Restricted individual-b SEs/ACI are **unsupported**
throughout, including inactive sample constraints, to avoid a data-dependent
switch of methods without a uniform justification. Unrestricted zero-count
fits similarly lack full covariance or individual-b/probability Wald inference.

The time likelihood is unchanged by the restriction and by cause counts at
zero. Its (a,B) covariance is supported separately under regular **time-model**
asymptotics even when cause inference is nonregular. This is not inversion of
the Hessian along an active face. All-m_i=1 experiments have no finite shape
MLE; failed/nonfinite/nonstationary fits cannot be passed as valid estimates.

### I5. Reliability and all analytical gradients

At an interior evaluation time 0<t<1 define L=log(1-t^a), P=dL/da,
S=exp(BL), F=1-S, h=aBt^(a-1)/(1-t^a),
v=(1-2^(-1/B))^(1/a). v is **median** lifetime (the project's MTF label),
not mean time to failure. Point estimates reuse the existing Stage 1 models;
zero-cause cases use the exact continuous closure pF or (1-p)F.
In local (a,B,p) coordinates:

\[
\begin{array}{ll}
\nabla S=(SBP,SL,0),&
\nabla h=h(1/a+\log t-P,1/B,0),\\
\nabla v=v(-\log v/a,-\rho/(aB),0),&
\rho=z_m/(e^{z_m}-1),\quad z_m=\log2/B,\\
\nabla p=(0,0,1),&\nabla(1-p)=(0,0,-1),\\
\nabla(pF)=-p\nabla S+(0,0,F),&
\nabla((1-p)F)=-(1-p)\nabla S-(0,0,F).
\end{array}\tag{I6}
\]

For the median, log v=log(1-exp(-log2/B))/a; differentiating in B gives
-log2 exp(-log2/B)/[a B²(1-exp(-log2/B))], explaining the negative sign.
For the hazard, differentiate its log to obtain the displayed formula.
Use (1-aP)/a+log t rather than subtracting nearly equal 1/a and P.
For incidence, the product rule applied to p(1-S) gives the last line.
Embed each gradient in the full block parameter vector with zeros for other
blocks and compute Var(ghat) approximately grad(g)^T V grad(g).
For S,h,v only the time covariance is needed. Use log Wald intervals for h,
and logit Wald intervals for bounded S,v and (regular unrestricted) incidence.
Probabilities retain exact binomial intervals; their SEs are supplied only
for regular unrestricted fits. Reliability intervals at multiple t are
**pointwise marginal**, not simultaneous bands.

Restriction leaves a,B and hence S,h,v and their time-based uncertainty
unchanged, including when p is active. Incidence depends on p; its ordinary
delta uncertainty is provided only for regular unrestricted interior fits.
Restricted/zero-boundary incidence uncertainty is explicitly unsupported.
Numerically saturated point estimates at 0 or 1 cannot support logit delta
intervals; the result explicitly marks such intervals unavailable.

### I6. Numerical verification and interfaces

`src/uncertainty.py` exposes `observed_information`,
`covariance_from_information`, `parameter_inference`, `binomial_interval`,
`reliability_estimates`, `reliability_gradients`, and `reliability_inference`.
Information includes ordered coordinate names; covariance outputs use those
orders. Parameter reports carry time/full/original covariance separately,
per-parameter SE/interval dictionaries and numerical/restriction diagnostics.
A missing SE or empty interval dictionary means unsupported or failed;
inspect diagnostics. Exact empty sets use Interval.status='empty', rather
than the unavailable status for numerical interval failure.
The fit/data score checks prevent silently applying a fit from another model
or dataset. The routine does not refit or modify supplied estimators.

`tests/test_uncertainty.py` checks independent central second differences of
the existing Stage 3 likelihood for 1–3 blocks in both parameterizations,
nonlinear Hessian correction away from the fit, transformed covariance against
an independent original-matrix solve, SE and interval formulas, Cholesky and
conditioning failures, exact set coverage by binomial enumeration, zero-count
and active/inactive constraints, finite-difference reliability gradients and
independent delta quadratic forms. Deterministic enumeration verifies the
chosen finite cases, while I4 provides the general coverage argument.
`python -m experiments.classical_inference_example` prints the two-block
teaching analysis and explicitly flags unsupported inference.

### I7. Scope and limitations

These are classical uncertainty additions and deterministic validation, not
a Monte Carlo coverage study or a reproduction of a published result table.
The two-block data have only seven failures, so asymptotic normality is
particularly tentative. Compatible published numerical reproduction remains
a separate unfinished validation goal. No expected-information integration,
profile-likelihood interval, simultaneous band, restricted individual-b/CIF
interval, Bayesian method, across-block order, new simulator, or real-data
analysis is introduced. Existing Stages 1–4 and restricted MLE code/tests are
preserved; reference inconsistencies are handled in this new derivation.

Recorded verification on 2026-10-26: `.venv/bin/python -m pytest -q`
reported **309 passed in 1.44 s** (254 preserved tests plus 55 new tests).
All 13 preexisting source/test files matched their pre-task SHA-256 hashes;
no existing estimator or test was edited. `git diff --check` passed.
For the teaching fit, independent central likelihood differences with absolute
step 10^(-5) gave maximum Hessian-entry errors 1.485e-6 (original coordinates)
and 1.827e-6 (total/proportion coordinates). The transformed original covariance
agreed with an independent direct linear solve to maximum absolute error
5.552e-17. Finite-difference errors depend on step and cancellation; the tests
also use separate moderate, off-fit points with explicit tolerances.

The teaching analysis prints the following illustrative values (95% intervals):

| Quantity | Estimate | SE | Interval / method |
|---|---:|---:|---|
| Common a | 1.173960 | 0.542361 | [0.474684, 2.903369], log Wald |
| B_A | 0.776562 | 0.554363 | [0.191660, 3.146456], log Wald |
| B_B | 0.715014 | 0.438624 | [0.214855, 2.379485], log Wald |
| Restricted p_A | 0.500000 | unsupported | [0.094299, 0.500000], exact intersection |
| Restricted p_B | 0.250000 | unsupported | [0.006309, 0.500000], exact intersection |
| S_A(0.5), both fits | 0.634626 | 0.168675 | [0.294498, 0.878453], logit delta |
| h_A(0.5), both fits | 1.451328 | 0.889990 | [0.436303, 4.827734], log delta |
| Median_A, both fits | 0.638353 | 0.194611 | [0.252750, 0.902071], logit delta |

These are computations on hand-constructed observations, not experimental
coverage results or published numerical values. Restricted individual-b and
incidence SEs/intervals are not supplied even when their point values exist.
