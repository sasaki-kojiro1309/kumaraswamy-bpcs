# Verified Bayesian derivations: Kumaraswamy BPCS

Date: 2026-11-01. This document concerns exactly the existing common-a,
block-specific-(b_i1,b_i2), known-cause, fixed completed BPCS model. There is
no ordering between facilities, missing-cause likelihood or hybrid stopping.
The classical code, data contracts and estimator algorithms are unchanged.

## Sources and verification scope

Starting source: the supplied **Group_22-4.pdf**, Sections 4.1–4.4,
PDF pp.8–13, equations (16)–(35). Found at
`/Users/mac_m5/Downloads/Group_22-4.pdf`; a byte-identical copy is retained at
`References/Group_22-4.pdf`, SHA-256
`68685927ad631524eaf5ac56eed8a60287019a490507f79fff52b9ef5a30808b`.
All 14 pages were text-extracted; the Bayesian sections were read and the
propriety, conditional, restriction and interval pages 9,10,12,13 rendered and
visually inspected. This supplied report is not relabeled as a published paper.
No change is made to Report.tex, the original report or the three references.

Preserved literature labels: **Reference 1 — Wang et al. (2023)** supplies the
block progressive structure for IEED; **Reference 2 — Wang (2018)** supplies
Kumaraswamy competing-risk mathematics for progressive Type-II sampling;
**Reference 3 — Mahto et al. (2022)** concerns partial causes and generalized
hybrid stopping, which we do not adopt. Equations below follow our verified
likelihood, not a direct transfer of any other distribution or design.

For computation, [Neal's slice-sampling paper](https://arxiv.org/abs/physics/0009028)
supports stepping out and shrinkage. Rank-normalized/folded split Rhat and
bulk/tail ESS follow the constructions in
[Vehtari et al.](https://arxiv.org/abs/1903.08008) and their
[author-hosted appendix](https://avehtari.github.io/rhat_ess/rhat_ess.html).
These diagnostics are checks, not mathematical certificates of convergence.

## B1. Shared likelihood and signs

Write w_ij=1+r_ij, y_ij=-log x_ij>0, y_i*=min_j y_ij,
B_i=b_i1+b_i2, d_is=number of observed cause-s failures, m_i=d_i1+d_i2,
M=sum_i m_i, T=sum_ij log x_ij,
H_i(a)=sum_j log(1-x_ij^a), W_i(a)=-sum_j w_ij log(1-x_ij^a)>0.
With C the existing product of pre-failure risk sets,

\[
\ell=\log C+M\log a+(a-1)T+\sum_{is}d_{is}\log b_{is}
       -\sum_iB_iW_i(a)-\sum_iH_i(a). \tag{B1}
\]

A cause-s failure contributes f_is S_i,other; a withdrawn survivor contributes
S_i,minimum. Their product gives exponent w_ij B_i-1 on (1-x_ij^a).
Thus the survival penalty -B_i W_i is negative, whereas -H_i is positive.
There is no additional cause-count multinomial factor for the recorded sequence.
`independent_logposterior`, `hierarchical_logposterior`, and the hierarchical
a-conditional call the existing `src/likelihood.py` implementation.
The conjugate summaries use Stage 1 log survival, with log-sum-exp for W_i;
no inconsistent replacement likelihood is introduced.

Completion optimization, 13 November 2026: the hierarchical a update now uses
the B8 sufficient-statistic kernel directly. Terms constant in a are omitted
only inside that conditional slice update. Regression tests compare its log
ratios with the original row-wise likelihood plus Gamma prior. The public joint
log-posterior and posterior target remain unchanged.

## B2. Independent scale prior

The prior is pi_I proportional to a^(-1) product_is b_is^(-1). It is improper,
and is an independent scale prior, not a claim of a joint Jeffreys prior.
Multiplying (B1) gives

\[
\pi_I(a,b\mid D)\propto a^{M-1}
\prod_{ij}\frac{x_{ij}^{a-1}}{1-x_{ij}^a}
\prod_{is}b_{is}^{d_{is}-1}\exp[-\sum_iB_iW_i(a)]. \tag{B2}
\]

All Gamma laws here use **shape-rate**: density
r^q/Gamma(q) v^(q-1) exp(-rv). For fixed a, positive counts give independent
b_is|a,D ~ Gamma(d_is,W_i(a)). Integrating each b uses
integral_0^infinity b^(d-1)exp(-Wb) db=Gamma(d)W^(-d). Hence, dropping only
factors constant in a,

\[
h_I(a)=a^{M-1}\exp[(a-1)T-\sum_iH_i(a)]
             \prod_iW_i(a)^{-m_i}. \tag{B3}
\]

The logarithm equals (M-1)log a+(a-1)T-sum H_i-sum m_i log W_i.
The transformed slice target in eta=log a includes +eta for da/deta=a.
This is the marginal target, not the full conditional given current b.
Constants from the arbitrary scale of the improper prior do not define a
marginal likelihood for model comparison.

### Propriety

If any d_is=0 in the unrestricted model, integration of b_is^(-1)exp(-W_i b_is)
at zero diverges. Otherwise let
D=sum_ij(y_ij-y_i*) >=0. Near zero,
1-x_ij^a ~ a y_ij, W_i(a)=n_i log(1/a)+O(1), so

\[
h_I(a)\asymp\frac{1}{a\{\log(1/a)\}^{M}},\quad a\downarrow0.
\]

This is integrable exactly when M>1. At infinity,
W_i(a)~w_i,last exp(-a y_i*) and H_i(a) tends to zero, so

\[
h_I(a)\asymp a^{M-1}e^{-Da},\quad a\to\infty. \tag{B4}
\]

Consequently positive counts, M>1 and D>0 establish propriety. If D=0 the
upper integral diverges. Positive counts imply m_i>=2 for the supported data,
but the API still checks all conditions explicitly. D is evaluated with log1p
ratios for nearly identical times so subtracting rounded logs does not erase
a positive gap. No independent-prior sampler runs outside this verified domain.

### Moments

All positive moments of a are finite when (B4) has D>0. For r>0,
E[b_is^r|a,D]=Gamma(d_is+r)/Gamma(d_is) W_i(a)^(-r).
The upper-tail integrand for this moment is proportional to
 a^(M-1) exp[-a(D-r y_i*)]. At equality its polynomial integral also diverges;
the lower endpoint is integrable with an extra log(1/a)^(-r). Therefore

\[
E[b_{is}^r\mid D]<\infty\quad\Longleftrightarrow\quad D>r y_i^*. \tag{B5}
\]

We check r=1 before returning a squared-error Bayes estimate and r=2 before
presenting a mean MCSE. A finite sample average is not a moment-existence proof.
If the first moment is infinite, its estimate is None, even if the chain
passes diagnostics. Properness still permits credible intervals.

## B3. Within-block restriction for the independent prior

Change variables b_i1=B_i p_i, b_i2=B_i(1-p_i), with absolute Jacobian B_i.
For one block the kernel becomes

\[
B_i^{m_i-1}e^{-W_iB_i}\,p_i^{d_{i1}-1}(1-p_i)^{d_{i2}-1}. \tag{B6}
\]

Thus B_i|a,D ~ Gamma(m_i,W_i(a)), p_i|D ~ Beta(d_i1,d_i2), and p_i is
independent of a and all totals under the joint posterior. The order is exactly
p_i<=1/2. Its probability is I_(1/2)(d_i1,d_i2), independent of a, and the
combined event probability is the product over blocks. Conditioning changes
only the Beta proportions, now truncated to (0,1/2]; it leaves the entire a
marginal and conditional/unconditional total distributions unchanged.
All minimum-survival, hazard and median posterior distributions are consequently
unchanged too, under this prior only. Cause probabilities and incidence change.

The sampler first updates eta from (B3), then independently draws Gamma totals
and unrestricted/truncated Beta proportions and reconstructs both b values.
Separate RNG streams couple a and totals exactly across the two teaching runs;
this is a reproducibility device, not the proof of equality. The truncated Beta
inverse CDF uses U*I_(1/2)(d1,d2). If the event probability or reconstructed
positive values cannot be represented, computation fails explicitly.
Posterior means are not projected, sorted or pooled to enforce order.

For reference E[p_i|p_i<=1/2]=d_i1/m_i times
I_(1/2)(d_i1+1,d_i2)/I_(1/2)(d_i1,d_i2). Combining with E[B_i|a]
verifies the conditional restricted b means and their same moment criterion.

### Qualification about zero counts

The request's blanket statement that ordering cannot rescue **any** zero-count
impropriety is too broad. If d_i1>0,d_i2=0, (B6) has p density
p^(d_i1-1)/(1-p) on (0,1/2], which is integrable. The total integral is also
finite for fixed a. For example two ordered failures of cause 1 at .2 and .8
can have a proper restricted posterior although their unrestricted posterior
is improper. If d_i1=0, the singularity at p=0 remains and restriction does not
rescue it. **Our independent-prior API nevertheless conservatively requires
both counts positive, exactly as requested.** Restricted zero-count cases are
`unsupported_zero_count`, with propriety=None and moments unverified, rather
than an incorrect blanket assertion of mathematical impropriety. Unrestricted
zero-count cases are labeled improper. There is no silent prior substitution.

## B4. Hierarchical Gamma model and all conditionals

Let b_is|kappa_s,lambda_s ~ Gamma(kappa_s,lambda_s), independently conditional
on the two shared hyperparameter pairs. Set independent proper priors
 a~Gamma(A_a,B_a), kappa_s~Gamma(A_s,B_s), lambda_s~Gamma(C_s,D_s).
Every fixed shape/rate is positive and is a configuration input. Write k for
the number of blocks, avoiding confusion with kappa. Multiplication gives

\[
\begin{split}
\pi_H&\propto a^{M+A_a-1}e^{-B_aa}
\prod_{ij}\frac{x_{ij}^{a-1}}{1-x_{ij}^a}\\
&\quad\times\prod_{s=1}^2\left\{
\frac{\kappa_s^{A_s-1}e^{-B_s\kappa_s}
\lambda_s^{k\kappa_s+C_s-1}e^{-D_s\lambda_s}}
{\Gamma(\kappa_s)^k}
\prod_i\left[b_{is}^{\kappa_s+d_{is}-1}
 e^{-(\lambda_s+W_i(a))b_{is}}\right]\right\}. \tag{B7}
\end{split}
\]

In particular lambda_s^(k kappa_s) and Gamma(kappa_s)^(-k) **must not be
omitted**: both depend on unknown state variables. The implementation includes
even the fixed hyperprior normalizers and log C; it still returns an unnormalized
joint posterior, not a normalized posterior density or evidence.

Collecting terms in each coordinate yields

\[
\begin{split}
b_{is}\mid rest,D&\sim\mathrm{Gamma}(\kappa_s+d_{is},\lambda_s+W_i(a)),\\
\lambda_s\mid rest,D&\sim\mathrm{Gamma}(k\kappa_s+C_s,D_s+\sum_i b_{is}),\\
\log\pi(\kappa_s\mid rest,D)&=(A_s-1)\log\kappa_s-B_s\kappa_s
-k\log\Gamma(\kappa_s)
+\kappa_s\{k\log\lambda_s+\sum_i\log b_{is}\}+const,\\
\log\pi(a\mid rest,D)&=(M+A_a-1)\log a-B_aa+(a-1)T
-\sum_iH_i(a)-\sum_iB_iW_i(a)+const. \tag{B8}
\end{split}
\]

Gibbs draws b and lambda from their Gamma laws. Slice updates use xi_s=log kappa_s
and eta=log a and add the respective +xi_s/+eta Jacobian. No Gamma assumption
is made for either nonstandard shape conditional. The common-a target calls
the existing likelihood; b and hyperparameters remain fixed during that update.
A tiny-shape Gamma draw uses the exact identity Gamma(q)=Gamma(q+1)*U^(1/q),
evaluated on the log scale, not a replacement of q by a positive epsilon.

### Within-block restricted hierarchy

Our restricted distribution is the **entire joint posterior conditioned on**
C={b_i1<=b_i2, all i}. It is not a different prior that independently normalizes
each truncated b distribution conditional on the hyperparameters. That latter
choice would add hyperparameter-dependent order-event denominators and change
the target. Here each b_i1 update is (B8) truncated to (0,current b_i2), and
each b_i2 update is truncated to (new b_i1,infinity). All other full conditionals
remain (B8), since the event indicator is fixed during their updates.
Every recorded state satisfies order; there are no cross-block bounds.

For unequal cause-specific rates r1=lambda1+W_i(a), r2=lambda2+W_i(a), the
conditional event probability is I_(r1/(r1+r2))(kappa1+d1,kappa2+d2), which
in general depends on a and the hyperparameters. There is **no hierarchical
invariance theorem** for a, totals or reliability. Classical pooling and the
independent-prior Beta/Gamma simplification are not used for this sampler.

### Propriety without assuming likelihood bounded uniformly in a

Let L_max(a)=sup_b L(a,b). Even when a cause count is zero, this finite supremum
at fixed a is obtained by the existing d/W formulas on the appropriate closure.
From the verified profile derivation,
 L_max(a) asymptotic to const*{log(1/a)}^(-M) at zero and
 const*a^M exp(-Da) at infinity, including D=0.
Integrate the normalized conditional prior for b at fixed (kappa,lambda), then
the normalized hyperpriors. Since L<=L_max(a), the posterior normalizer is
bounded by integral L_max(a) Gamma(a;A_a,B_a) da. It is finite at zero for A_a>0
and at infinity for B_a>0, even if all blocks have one failure. The integrand is
strictly positive on an open set, so the normalizer is nonzero. Restricting to
C, which has positive prior/posterior probability, preserves finite positive
normalization. This establishes propriety for **all valid data and all positive
fixed Gamma hyperparameters**, including zero counts. A proper prior alone,
without this likelihood bound, would not have established it.

### Sufficient moment bounds

Multiplying by a^r leaves the preceding integral finite for all r>0. Likewise,
any positive kappa/lambda moment is finite because L_max(a) is independent of
those coordinates and their proper Gamma hyperpriors have all positive moments.
For b_is^r, maximize b_is^r L over b: replace the selected count by d_is+r
in its supremum while the likelihood still has a^M. Near zero this bound adds
{log(1/a)}^(-r); at infinity it adds exp(r a y_i*). Thus

\[
B_a+D>r y_i^*\ \Longrightarrow\ E_H[b_{is}^r\mid D]<\infty. \tag{B9}
\]

This verifies the supplied report's **sufficient** first-moment condition and
extends it to r=2 for MCSE. Failure of (B9) is marked unverified, not infinite.
Finite moments remain finite after conditioning on C of positive probability.
No claim of necessity is made for hierarchical b moments.

## B5. Reliability means and moment checks

For every joint posterior draw compute the existing Stage 1 S_i(t), h_i(t)
and median with B_i=b_i1+b_i2, and compute p_i=b_i1/B_i,
G_i1(t)=p_i[1-S_i(t)], G_i2(t)=(1-p_i)[1-S_i(t)]. Average these values,
not nonlinear functions evaluated at mean parameters. Credible intervals also
use their own draw-wise values. Mission times are fixed interior times; key
names retain the input float precision so nearby mission times are not merged.
These are marginal, pointwise credible intervals, not simultaneous bands.

Survival, probabilities, incidence and the median are bounded in [0,1], so
all moments exist for every proper posterior. A hazard is unbounded and needs
its own check. Put u_t=-log t>0. Then
 h_i(t)=B_i a/[t(exp(a u_t)-1)]. This is at most B_i/(t u_t), tends to that
bound near a=0, and is asymptotic to (a/t) B_i exp(-a u_t) at infinity.
For the independent prior (including its restriction), the conditional Gamma
moment in (B6) gives the **exact** hazard moment condition

\[
E_I[h_i(t)^r\mid D]<\infty
\quad\Longleftrightarrow\quad D>r(y_i^*-u_t). \tag{B10}
\]

At equality the upper integral contains a^(M+r-1) and diverges. At zero the
extra inverse-log factor is integrable. For the hierarchy, the analogous
supremum bound proves the **sufficient** condition
 B_a+D>r(y_i*-u_t), with failures unverified. Hence an infinite B mean does
not by itself establish an infinite hazard mean. Both r=1 and r=2 are checked
for every mission-time hazard. Mean MCSE is withheld if a finite second moment
is not established. No finite Monte Carlo average overrides any moment check.

## B6. Numerical kernels, diagnostics and intervals

The slice sampler randomly positions a width-w bracket, steps each end out
until its density is below the slice height, then uniformly proposes and shrinks
around the old state. This is the full stepping-out construction, not a
silently shortened bracket after a failed budget. If either stepping out or
shrinkage exhausts its budget, sampling fails explicitly. Log-coordinate guards
are computation limits, not posterior truncation bounds: contacting them aborts
the computation; no estimate is certified from the stopped chain.

Truncated Gamma draws invert a conditional CDF or survival probability, choosing
the tail to avoid subtraction near one. Log-difference calculations retain
small interval probabilities. Far beyond SciPy's ordinary CDF/SF range, a
convergent lower-incomplete-Gamma series or upper continued fraction evaluates
log probabilities, and a bracketed scalar root in log x inverts them. There is
no rejection loop with a vanishing order-event acceptance rate. Unresolved
probabilities, nonrepresentable quantiles, or values outside the strict interval
produce explicit numerical failure, with no epsilon repair or endpoint clipping.

Independent SeedSequence streams produce reproducible, distinct chains.
Diagnostics use split chains, pooled normal scores Phi^(-1)((rank-3/8)/(N+1/4)),
and the maximum of rank-normalized and folded rank-normalized split Rhat.
Bulk ESS uses the rank scores; tail ESS is the minimum ESS of the 5% and 95%
quantile indicators. Multi-chain FFT autocovariances use Geyer's initial-positive
and monotone paired sums. ESS is capped at sample size, a conservative choice
for negatively autocorrelated chains. Raw-mean ESS supports MCSE=sample SD/sqrt(ESS)
only with a verified finite second moment. MCSE additionally assumes adequate
mixing and an applicable MCMC central-limit approximation; a general CLT is
not proved here. Scaling makes means/autocovariances
safe for large finite samples; it does not change the target or repair infinity.

Defaults are four chains, 1000 warm-up iterations, 2000 retained draws per chain,
rank Rhat<=1.01 and bulk/tail ESS>=400 for **every parameter and requested
reliability quantity**. Thresholds and budgets are explicit configuration inputs.
`diagnostics_passed` is not proof of mixing or a guarantee against missed modes;
inspect traces and scientific plausibility too. Failed checks withhold **all**
posterior-mean estimates; draws/intervals are explicitly exploratory. A numerical
failure is distinct from an improper/unsupported prior and from failed diagnostics.

For interval mass q and N retained pooled samples, set c=ceil(qN). Sort x and
minimize x[j+c-1]-x[j] over j=0,...,N-c. Return the closed endpoints and the
**actual** fraction of observations between them (ties can increase it).
The supplied report's floor(qN) index offset includes floor(qN)+1 points;
when qN is an integer it includes one unnecessary point. Our indexing uses
exactly the smallest count attaining the requested fraction. The method is a
**shortest contiguous sample interval**. It approximates an HPD interval for
suitable unimodal marginals; multimodal HPD sets may be disconnected. HPD is
not invariant under nonlinear reparameterization. Restricted intervals come
from restricted draws, not clipped unrestricted intervals.

## B7. Validation and remaining limits

Tests independently expand the existing likelihood; compare the hierarchical
joint with normalized SciPy Gamma densities; verify joint/full-conditional
ratios for every coordinate; integrate Gamma kernels; verify the B,p Jacobian
and rate-independent order probabilities; and compare independent-a means with
one-dimensional quadrature over log a using the proved lower-tail expansion.
Both nonstandard slice targets have separate quadrature checks. Fixed-seed
sampling tests use Monte Carlo tolerances, not exact random-value snapshots.

Gamma truncation tests cover ordinary truncation, a right tail starting at 1000,
a lower-tail Gamma(100,1) cut at .001, and the narrow interval
(1000,1000.00001). Probability integral transforms use conservative DKW-scale
bands, and exact Gamma(2) tail identities plus independently scaled quadrature
check log-tail calculations. Diagnostics are checked on iid, AR(1), shifted,
scale-mismatched and stuck chains. Tests also cover zero counts, infinite and
unverified moments, arbitrary blocks, multiple seeds, ordering without any
facility order, shortest-interval mass/indexing, reliability averaging and
numerical/convergence failure reporting. See BAYESIAN_GUIDE.md for executed results.

No empirical bias/coverage claim or published table reproduction is made.
The hyperpriors in the teaching example are illustrative; they need elicitation
and later sensitivity analysis. Heavy tails can need much longer chains than
the example. Finite range, slice budgets, underflowing truncated-Beta event
probabilities and unrepresentable small-shape Gamma draws can stop computation.
The independent prior's restricted zero-count cases remain deliberately outside
the supported domain even when one could prove their propriety separately.
Hierarchical b/hazard moment conditions are sufficient only. No simulation
study, real-data application, across-block/combined order or new loss is added.
