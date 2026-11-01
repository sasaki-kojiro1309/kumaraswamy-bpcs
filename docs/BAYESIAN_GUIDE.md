# Bayesian estimation guide

The API provides four analyses on the same validated BPCS experiment:
independent scale prior or hierarchical Gamma prior, each unrestricted or with
b_i1<=b_i2 separately within every block. No ordering between facilities is
imposed. The classical implementation is preserved. The supplied report's
Bayesian mathematics is verified in BAYESIAN_DERIVATIONS.md; that document
also explains the conservative zero-count policy and sufficient moment bounds.

## Run the four-method teaching example

```sh
.venv/bin/python -W error -m experiments.bayesian_example --output results/bayesian_teaching_example.json
.venv/bin/python -m pytest -q -W error
```

This uses the existing fixed seven-failure, two-block data. It is not a BPCS
simulation, coverage experiment or real-data analysis. The JSON records the
input rows, priors, sampler controls, diagnostics, moments, intervals and means.
The configured prior is illustrative, not an elicited scientific prior:
 a~Gamma(2,2), kappa_1,kappa_2~Gamma(3,2), lambda_1,lambda_2~Gamma(2,2).
The second argument is always a **rate**, not a scale. Change these explicitly
when there is a scientific basis; no undocumented prior constants are hidden.

## Python interface

```python
from experiments.stage2_example import load_example
from src.bayesian import GammaPrior, HierarchicalPrior, SamplingConfig, fit

data = load_example()
config = SamplingConfig(chains=4, warmup=1000, draws=2000, seed=20261101)
hyperprior = HierarchicalPrior(
    a=GammaPrior(shape=2, rate=2),
    kappa=(GammaPrior(3, 2), GammaPrior(3, 2)),
    lam=(GammaPrior(2, 2), GammaPrior(2, 2)),
)

independent = fit(data, prior='independent', config=config, times=(.25, .5))
independent_ordered = fit(data, prior='independent', restricted=True, config=config)
hierarchical = fit(data, prior=hyperprior, config=config)
hierarchical_ordered = fit(data, prior=hyperprior, restricted=True, config=config)

print(independent.status, independent.warnings)
if independent.converged:
    print(independent.estimates['a'])
    print(independent.moments['b1[A]'])
    print(independent.intervals['b1[A]'])
    print(independent.reliability['A']['survival@0.5']['estimate'])
    print(independent.diagnostics['parameters']['a'])
```

`prior` and `restricted` select all four methods; future comparison code can
pass the exact same data object to each. `GammaPrior`, `HierarchicalPrior`
and `SamplingConfig` validate configuration immediately. Original Stage 2
constructors continue to validate data. There are no new runtime dependencies.
Public raw kernels and conditional parameters are also available for verification:
`independent_logposterior`, `independent_log_marginal`,
`independent_conditionals`, `hierarchical_logposterior`,
`hierarchical_conditionals`, `kappa_logconditional`, and `a_logconditional`.
Gamma conditional rates are returned as log rates where necessary for stability.
`posterior_conditions` can be called before any sampling.

## Understand the computation

With the independent prior, sampling needs just one nonstandard coordinate:
a's marginal posterior. Given a, draw a Gamma total B_i and a Beta cause share
p_i, and reconstruct b_i1=B_i p_i, b_i2=B_i(1-p_i). Restriction truncates only
p_i to (0,1/2]. Thus a and the minimum-lifetime reliability distribution are
unchanged under this prior. Posterior b means are not classical pooled MLEs.

With the hierarchy, conditional Gamma draws update b and lambda; log-scale
slice updates handle a and kappa with the correct transformation Jacobians.
Restricted b updates use Gamma conditional distributions truncated against
the other cause's current value. The a marginal and totals generally change;
the independent-prior invariance is not extended to this model.

Survival, hazard, median, cause probabilities and cumulative incidence are
calculated for **every joint draw** through the established Stage 1 model.
Averaging those functions estimates their posterior means. Applying a nonlinear
function to posterior mean parameters generally gives a different quantity.
MTF continues to mean median lifetime, not mean lifetime.

## Result structure and validity

`BayesianResult` includes:

- `method`, `restricted`, `status`, `converged` and `configuration`.
- `names` and `draws`, with shape (chains, retained draws, parameters).
  Order is a, then b1/b2 for each block in data order, then (for the hierarchy)
  kappa1,kappa2,lambda1,lambda2. Names retain block IDs.
- `estimates`: squared-error posterior means, or None when unavailable.
- `moments`: per-parameter first/second moment classifications: finite,
  infinite, unverified, or undefined for an improper posterior.
- `intervals`: shortest contiguous sample intervals, with requested probability,
  actual empirical mass and a qualified method label.
- `diagnostics`: rank-normalized/folded split Rhat, bulk/tail ESS, mean ESS,
  mean MCSE where finite variance is verified, propriety conditions and elapsed time.
- `reliability`: per block/quantity estimates, intervals, moments, diagnostics
  and joint-draw function arrays. Mission-time keys use round-trip float strings.
- `warnings`: explicit moment, diagnostic and numerical limitations.

The independent prior requires every cause count positive, M>1 and D>0.
Unsupported/improper independent cases raise `PosteriorError` **before sampling**.
Some restricted zero-cause-2 cases could be mathematically proper, but remain
conservatively unsupported by design; `posterior_conditions` returns
propriety=None for these rather than calling every such posterior improper.
The hierarchy's proper Gamma hyperpriors establish posterior propriety even
with zero counts or single-failure blocks; this does not imply every b mean
is finite. Check the sufficient bounds in the derivation.

For the independent model, a b mean is finite exactly when D>y_i*, and its
variance is finite exactly when D>2 y_i*. For the hierarchy,
B_a+D>r y_i* is sufficient for order-r b moments; failures are unverified.
Mission-time hazard moments have separate, sharper checks. Bounded survival,
median, probabilities and incidence have finite moments under a proper posterior.
No finite Monte Carlo average overrides these checks. `converged=True` can
coexist with None estimates for infinite/unverified moments: convergence and
moment existence answer different questions.

`diagnostics_passed` requires Rhat<=1.01 and bulk/tail ESS>=400 by default for
all parameters **and requested reliability quantities**. These are configurable
checks, not proof of convergence. Examine traces (available in `draws`) and
scientific plausibility; heavier tails or complex priors can need longer runs.
`diagnostics_failed` withholds all means and marks available draws/intervals as
exploratory. A slice/truncation/range failure returns `numerical_failure` and
never certifies a fit. Mean MCSE is unavailable if a finite second moment is
not established; even with finite variance it is an asymptotic mixing-dependent
estimate, not a proven general MCMC central-limit theorem.

The interval is the shortest closed interval containing at least ceil(qN)
pooled retained observations; ties may increase its empirical mass. It is an
HPD approximation for suitable unimodal marginals. A multimodal HPD region may
be disconnected, and HPD is not transformation invariant. Restricted intervals
come from restricted draws, never by clipping unrestricted intervals. Credible
interval mass is posterior mass; it is not a claim of repeated-sampling coverage.

## Numerical limits and scope

Log posteriors, log W, log Gamma draws and tail-probability inversion avoid
ordinary probability products. No epsilon substitutes a zero parameter;
posterior draws are strictly positive. Very small Gamma shapes, very rare
truncated-Beta events or unrepresentable tails can still require numerical
failure. Log-coordinate guards and budgets abort computation rather than
silently truncate the posterior. There is no rejection sampler with vanishing
order-event acceptance probability. Exact reproducibility assumes the same
NumPy/SciPy versions, seed and configuration; elapsed times naturally vary.

Hierarchical b/hazard moment failures remain unverified, not declared infinite.
Independent restricted zero counts are intentionally excluded. Hyperprior
sensitivity, simulation performance, real-data analysis, across-block orders,
combined constraints and missing causes are outside this implementation.


## Executed validation (2026-11-01)
Baseline: 325 classical tests passed before implementation. Final command
`.venv/bin/python -m pytest -q -W error` reported **397 passed in 37.67 s**,
zero failures and zero skips: 325 preserved cases plus 72 Bayesian checks.
All six classical example/verification modules also returned exit code 0 with
no stderr. Compilation and `git diff --check` passed. Classical source modules,
existing test files, Report.tex and the original reference PDFs were unchanged.

The four-method example used the exact recorded seed, priors and 4x2000 retained
draw configuration above; all methods passed diagnostics for parameters and
reliability quantities. Results from the executed JSON:

| Method | Mean a | MCSE of mean a | Largest Rhat, all quantities | Smallest bulk/tail ESS |
|---|---:|---:|---:|---:|
| independent_unrestricted | 1.08559882 | 0.00616921 | 1.000794 | 4586.2 |
| independent_within_block | 1.08559882 | 0.00616921 | 1.000794 | 4586.2 |
| hierarchical_unrestricted | 1.32941317 | 0.00796879 | 1.000813 | 3304.4 |
| hierarchical_within_block | 1.34842030 | 0.00959046 | 1.001009 | 2024.6 |

Independent one-dimensional quadrature gives E[a|D]=1.087392597729172.
The independent-prior teaching estimate differs by about 0.29 estimated MCSE.
Tests additionally compare two independent-prior seeds with quadrature, both
nonstandard shape slice conditionals with integration, and Gamma/Beta sampler
distributions with analytical identities and tail probability transforms.
Reproducibility is checked exactly for independent draws and for both
hierarchical restriction choices in the current environment.

The equal independent-prior a means arise from its proved common marginal and
coupled RNG streams. The hierarchical point estimates differ; no significance
or superiority claim is inferred from that Monte Carlo difference. These are
illustrative posterior calculations, not published-result reproduction or
frequentist performance findings.


The same one-dimensional integration checks every unrestricted independent-prior
b mean, not just the common a. For the executed 8000-draw example:

| Parameter | Quadrature mean | Monte Carlo mean | Estimated mean MCSE |
|---|---:|---:|---:|
| a | 1.08739260 | 1.08559882 | 0.00616921 |
| b1[A] | 0.49934068 | 0.50281614 | 0.00508156 |
| b2[A] | 0.24967034 | 0.25535277 | 0.00346114 |
| b1[B] | 0.17006185 | 0.17342536 | 0.00222800 |
| b2[B] | 0.51018556 | 0.51231117 | 0.00418090 |

All five differences are less than 1.65 estimated MCSEs in this teaching run.
This checks numerical posterior computation; it is not a frequentist coverage
or estimator-performance experiment. The final 72-case Bayesian suite includes
five posterior-mean quadrature comparisons, beyond the conditional and sampler
checks described above.
