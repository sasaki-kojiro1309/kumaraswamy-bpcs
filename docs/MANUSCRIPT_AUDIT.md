# Independent mathematical verification for Report.tex

Audit date: 2026-11-01. Phase A was completed before manuscript drafting.
Scope: all implemented classical estimation and uncertainty, not the unimplemented
Bayesian integration stub, across-block restrictions, simulation, or real data.
This audit does not adopt the previous audit's conclusions as proof.

## Evidence inspected

Reviewed src/model.py, censoring.py, likelihood.py, mle.py, order_restricted.py,
uncertainty.py, and bayesian.py (scope only); all seven test modules; all six
existing numerical demonstrations; the JSON teaching data and previous validation
artifact; DERIVATIONS, MODEL_SPEC, PROJECT_STATEMENT, BPCS_GUIDE,
WITHIN_BLOCK_VALIDATION, FINAL_AUDIT, RESEARCH_NOTES and Readme; Report.tex;
and the three references' model/design, likelihood, classical estimation and
restriction sections. Source PDFs were text-extracted; important reference
pages were checked visually in this or the preceding source review. This is
not an independent validation of every unrelated result in those papers.

References: Wang et al. (2023), Quality and Reliability Engineering International
39, 2736–2764, DOI 10.1002/qre.3382; Wang (2018), Journal of Computational and
Applied Mathematics 343, 719–736, DOI 10.1016/j.cam.2018.05.013; Mahto, Lodhi,
Tripathi and Wang (2022), Journal of Applied Statistics 49(8), 2064–2092,
DOI 10.1080/02664763.2021.1889999. Metadata checked in the actual PDFs.
The first uses IEED; the second includes common-shape single-block Kw; the
third uses generalized progressive hybrid censoring and partially observed causes.
Only appropriate structures are cited. No exact full multi-block model or
novelty claim is attributed to these sources.

## Assumptions checklist

These are assumptions or data/design contracts, not conclusions proved by tests.

| ID | Assumption / contract | Verification and limitation |
|---|---|---|
| A1 | Common fixed a>0 across blocks and causes | Model specification and every fitting interface agree. Scientific sharing assumption; not implied by independence or the block design. |
| A2 | Block/cause-specific positive b_ic | Original probability classes require positivity; estimator zeros represent limits only. No cross-block pooling assumed. |
| A3 | Independent latent competing times | Needed for product survival and subdensity f_c S_other. Not identifiable as a general latent-dependence claim from competing-risk records alone. |
| A4 | Independent units | Sampling assumption; exchangeability within a block used in risk-set argument. |
| A5 | Independent blocks | Joint likelihood multiplies; shared a does not induce random dependence. |
| A6 | Identical latent laws within a block | Required for risk-set exchangeability; not tested by a structurally valid table. |
| A7 | Blocks may differ in b | No equality, weighting or pooled-population law introduced. |
| A8 | Exactly two risks | Known cause codes 1,2; no third or masked cause model. |
| A9 | Correct known causes | BlockData rejects None. Misclassification/missingness unmodeled. |
| A10 | Fixed n_i,m_i and withdrawal counts | Scheme validation and all-small-schedules tests. Fixed counts do not imply deterministic survivor identities. |
| A11 | Uniform noninformative survivor selection | Conditional likelihood argument assumes this; observed counts alone cannot verify it. |
| A12 | Exact strictly increasing times within each block | Enforced; follows almost surely from continuous latent times, but rounded ties need a new measurement treatment. |
| A13 | Fixed support (0,1) | Enforced for observed times. An application needs justified predetermined physical bounds; dividing by observed maximum is not the assumed experiment. |
| A14 | Completion at failure m_i | n_i=m_i+sum r_ij and terminal risk set empty after removals; verified algebraically and in code. |
| A15 | Optional b_i1<=b_i2 separately per block | User's provisional scientific assumption; implies hazard/probability/latent stochastic orders, not an empirical ordering of counts. |

## Claim-by-claim verification checklist

"Proven" below means the stated conditional mathematical result has a complete
argument in the manuscript. "Approximate/conditional" does not claim a general
asymptotic theorem for arbitrary censoring sequences. Existing test names are
summarized by their mathematical role, rather than treating a passed test as a
proof. The new audit script supplies independent 100-digit checks.

| Claim | Assumptions and independent derivation | Code | Evidence | Status / limitation |
|---|---|---|---|---|
| Kw CDF, PDF, normalization, survival | a,b>0; differentiate F and substitute y=t^a | model.Kumaraswamy | test_model integral/Beta checks | Proven on open support; endpoint density is a code convention |
| Hazard and median | f/S; solve S(v)=1/2 | model.Kumaraswamy | calculus/median inversion | Proven; median is not mean |
| Endpoint limits | S(0+)=1,S(1-)=0; f~ab t^(a-1), f~ab[a(1-t)]^(b-1) | model support conventions | uniform/tail tests | Proven asymptotics; no point mass at endpoints |
| Minimum law and hazard sum | A1,A3; S_1 S_2=(1-t^a)^B | CompetingRisks.minimum | latent-product and hazard tests | Proven; not a mixture |
| Subdensities, probabilities, incidence | f_c S_other=(b_c/B)f_min; integrate | CompetingRisks.cause_* | subdensity integrals and sums | Proven; latent CDF differs from CIF |
| Independent minimum and cause | Subdensity ratio constant b_c/B | same | probability identities | Proven under shared a and latent model |
| Risk sets and completion | Subtract previous failures/removals; Y_j=sum_{l>=j}(1+r_l) | censoring.BlockScheme | exhaustive schemes n<=6 | Proven for fixed complete schemes |
| Correct density constant | Conditional next-failure density Y g(x) S(x)^(Y-1)/S(prev)^Y; telescope | likelihood.log_censoring_constant | sequential oracle, ordered-region integration | Proven for recorded ordered times/marks with identities marginalized |
| Withdrawal factors/full likelihood | Each withdrawn survivor contributes S(x); combine exponents | likelihood.* | latent/expanded/sequential comparisons | Proven under A4–A11; no extra mark multinomial coefficient |
| Full log likelihood and constants | Log of exact sampling density | likelihood.loglikelihood | manual uniform case and constants | Proven; include_constant=False omits only log C |
| Scores and W,H derivatives | P=u/(e^(au)-1), P'=-Q; direct differentiation | mle.score, uncertainty | finite differences, 100-digit profile checks | Proven finite interior parameters |
| Conditional b=d/W | Strict concavity for d>0; decreasing linear term for d=0 | mle.conditional_estimates | independent inner optimization | Proven; zeros are suprema, not positive MLEs |
| Profile and envelope derivative | Substitute conditional b; include -H(a) and m log W | mle.profile_* | independent 100-digit profile derivative | Proven; count-independent shape score |
| Strict concavity | W positive exponential series; uniform differentiated convergence on compact a>0; (log W)'' is rate variance; Q<1/a^2 | mle profile solver | numerical concavity/root tests; proof rederived | Proven in a, not asserted concave in log a |
| Finite maximizing a / unique closure pair | Lower tail -M log(-log a), upper tail M log a-aD; D>0 iff some m_i>=2 | mle.fit_mle | single/multiple failure tests | Proven under strict observed ordering |
| Positive unrestricted attainment | Every d_ic>0 plus an identifying block | mle status | boundary tests | Proven; otherwise finite closure optimum or unbounded design |
| All-single-failure unboundedness | D=0 upper-tail growth; along conditional B path | mle status | profile growth tests | Proven, no finite vector reported |
| Brent algorithm/convergence diagnostics | Unique root in a, sign preserved by log a | mle.fit_mle | remote starts/budget tests | Numerical method; finite budgets/range may fail |
| Direct optimizer verification | Independent joint log-parameter optimization | mle.fit_direct | teaching agreement, known-quantile tests | Numerical check, not performance study |
| B,p factorization | b1=Bp,b2=B(1-p), algebra; no parameter-density Jacobian | order_restricted | independent expanded objective | Proven |
| Restricted optimum and KKT | Maximize concave binomial term for 0<p<=1/2; upper/lower multipliers | order_restricted | raw and proportion KKT tests | Proven, exact zero only in closure |
| Shape and total invariance / likelihood loss | Restricted profile differs by count-only nonnegative penalty | order_restricted | profile-grid, survival and total tests; SLSQP | Proven for this restriction only |
| Restricted attainment | d_i1>0 in all blocks and some m_i>=2; d_i2=0 pools positively | order_restricted status | both zero-count directions | Proven; active order equality not zero boundary |
| Information in q | Independently differentiate all three score families | uncertainty.observed_information | 100-digit Hessians at six points; existing derivative tests | Exact observed negative Hessian, not expected information |
| Original information and nonlinear mapping | Chain rule J_theta=A^T J_q A - sum U_p Hess(p) | same | off-fit score correction tests | Proven; congruence alone valid at unrestricted stationary fit |
| Positive information at conditional optimum | D_B>0; time Schur complement equals -ell_profile''>0; p curvature>0 at interior p | same/covariance | Cholesky and covariance comparisons | Proven positivity; numerical conditioning still separate |
| Covariance/SE | Inverse observed information as regular large-sample approximation | parameter_inference | direct original-matrix solve | Approximate/conditional on identifiable regular time design, fixed k and sufficient increasing failures; no general design-specific CLT proved |
| Covariance Jacobian | Delta linearization b1=Bp,b2=B(1-p) | parameter_inference | independent matrix solve agreement | Approximate covariance, exact differential identity |
| Normal/log/logit Wald | Taylor derivative 1,1/v,1/[v(1-v)] | uncertainty._wald | formula/endpoints/failure tests | Approximate marginal; raw Wald may leave support |
| Binomial counts | Normalized time/mark density factors; sum marks with d successes | binomial_interval input model | mark-sequence marginalization | Proven for fixed m and noninformative selection |
| Clopper–Pearson | Invert binomial tails; beta quantile identities | binomial_interval | extremes and finite enumeration | Exact conservative marginal coverage in real arithmetic |
| Restricted exact set | For true p<=1/2, membership in C∩[0,1/2] equals membership in C | binomial_interval | finite enumerated coverage and empty sets | Proven; can be empty; no joint coverage claim |
| Restricted interval exclusions | ptilde can be nonregular; no ordinary constrained-b Wald supplied | parameter/reliability inference | active, inactive and zero tests | Supported scope verified, no missing procedure invented |
| Reliability gradients | Direct/log/product differentiation of S,h,v,p,pF | reliability_gradients | independent 100-digit differences for all 7 outputs | Proven differentials; float saturation remains possible |
| Delta SE/transformed intervals | Embed local gradients in common-a covariance; g^T V g | reliability_inference | independent quadratic-form tests | Approximate pointwise marginal, not simultaneous bands |
| Restricted minimum uncertainty invariant | a,B and time negative Hessian unchanged | time covariance functions | teaching covariance max difference 1.67e-16 | Proven algebraic invariance, numerical roundoff allowed |
| Restricted CIF/individual-b unsupported | Code returns no SE/CI for these, even inactive sample constraints | uncertainty | explicit None/unsupported tests | Scope restriction, not a claim impossible in principle |
| No Bayesian/real/simulation results | bayesian.fit raises NotImplementedError; no data generator or real dataset | bayesian stub and directories | inspected sources | Scope fact; broad project title is not a completed-result claim |

## Executed independent checks (Phase A)

Baseline full suite: **325 passed in 1.65 seconds**, `-W error` (no skips).
Added `experiments/manuscript_audit.py`, using standard-library Decimal at
100 digits and h=1e-20, with a row-expanded likelihood rather than production
score or Hessian formulas. Its results are in results/manuscript_audit.json:

- Maximum absolute negative-Hessian error: 1.4210854715202004e-14 over original
  and total coordinates at a=.35,1.3,4.5.
- Maximum absolute analytical-gradient error across all reliability outputs:
  1.4210854715202004e-14 at three varied parameter/time configurations.
- Maximum absolute profile-score error: 4.263256414560601e-14 at four a values.
- Profile/direct unrestricted log-likelihood error: 8.881784197001252e-16.
- Analytical/SLSQP restricted log-likelihood error: 1.9206858326015208e-14.
- Transformed original covariance/direct solve error: 5.551115123125783e-17.
- Restricted/unrestricted time covariance error: 1.6653345369377348e-16.
- Restricted exact coverage enumeration: minimum .9518794504168782 over
  m=1,3,10,40 and 101 equally spaced p values in [0,.5]. This finite check
  supports implementation; it does not replace the general set-membership proof.
- All six pre-existing example modules executed with warnings as errors,
  exit 0 and empty stderr; their stdout is retained in the JSON.

Source hashes in that artifact identify the audited inputs. Documentation
updates made later are recorded in Git; rerun the audit command to refresh the
hashes if needed. No production estimator or existing test was altered by the
audit phase because no confirmed mathematical/algorithm defect was found.

## Corrections identified before drafting (now applied)

1. MODEL_SPEC's opening status still called uncertainty deferred despite the
   implemented extension later in that same file. Correct this stale status;
   it is a documentation inconsistency, not a statistical formula correction.
2. Report.tex contains a nonexistent generic package, duplicate package/title
   commands, missing external bibliography, placeholder author names, and
   unrelated RA/entropy/simulation/real-data claims. Replace only this unsupported
   template material while preserving A4/11pt, margins, theorem environments,
   numerical citation convention and the supplied affiliation. Author names
   are not known and will not be invented.
3. Strengthen manuscript statements about regular asymptotics: no proof of a
   CLT for every possible triangular array of progressive schedules exists in
   the repository. Report Wald/delta results conditionally, and distinguish
   exact mathematical information identities from approximate inference.

## Phase B/C record

Report.tex was rewritten only after completing Phase A. Its ten main sections
follow the requested order, with two technical/reproduction appendices and a
three-entry, self-contained numerical bibliography. The A4 11pt article class,
0.9-inch horizontal / 1-inch vertical margins, theorem/proof environments and
numerical citations are preserved. Redundant/unused packages, the missing generic
package, duplicate title, nonexistent external bibliography and inherited
unrelated claims were removed. Placeholder author names were removed; actual
names remain unspecified and the supplied affiliation is retained.

The desktop compiler initially timed out with missing standard TeX resources.
The bundled Tectonic CLI was then run with approved cache/network access to
obtain the resources. The final cached build succeeded (exit 0), producing
output/pdf/Report.pdf: **17 A4 pages, zero TeX warnings**, including no overfull
or underfull boxes, missing glyphs, undefined labels or citations. All **76
source labels**, seven tables and three bibliography entries resolve. Numeric
results in the tables were checked against the actual inference outputs.

Final QA includes rendered inspection of all 17 pages, table and equation
placement, numbering, title/affiliation, mathematical symbols and bibliography.
An independent PDF character-bound check confirms every page's text lies within
the intended horizontal margins (2pt inspection tolerance); no blank or clipped
pages or question-mark references were found. The exact PDF and TeX SHA-256
hashes and page bounds are in results/manuscript_qa.json. The final compiler
log is results/manuscript_compile.log; build intermediates are kept under the
ignored tmp/pdfs directory.

Final full suite: **325 passed in 1.65 seconds**, no failures/skips, with
warnings treated as errors. Compilation of all Python files and git diff
--check passed. All 15 existing production/test Python files were compared
byte-for-byte against HEAD and remain unchanged. No existing test assertions
were weakened and no optimizer or scientific model assumption was altered.
The only new executable is a deterministic independent audit, not a new
estimator or inference procedure.

Remaining limits are scientific support/order justification, finite float64
range and conditioning, the absent general schedule-specific asymptotic proof,
unsupported restricted b/CIF intervals and simultaneous bands, and the
unimplemented Bayesian/simulation/real-data/published-table components. These
are distinguished from confirmed bugs. No unresolved mathematical contradiction
or compilation blocker was found in the completed scope. Author metadata is
the only intentionally unspecified manuscript field.
