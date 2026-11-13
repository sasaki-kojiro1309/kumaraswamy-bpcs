"""Independent algebra, quadrature, conditional distributions and MCMC checks."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, betaln, betainc
from scipy.stats import gamma, beta, kstest

from experiments.stage2_example import load_example
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.likelihood import loglikelihood, log_censoring_constant
from src.model import Kumaraswamy
from src.bayesian import (GammaPrior, HierarchicalPrior, SamplingConfig, PosteriorError,
    posterior_conditions, independent_logposterior, independent_log_marginal,
    independent_conditionals, hierarchical_logposterior, hierarchical_conditionals,
    kappa_logconditional, a_logconditional, fit, posterior_reliability, _gamma_draw)
from src.bayesian_sampling import (SamplingError, slice_step, truncated_gamma,
    _log_gamma_tail, shortest_interval, chain_diagnostics)


@pytest.fixture(scope='module')
def prior():
    return HierarchicalPrior(GammaPrior(2, 2), (GammaPrior(3, 2), GammaPrior(3, 2)),
                             (GammaPrior(2, 2), GammaPrior(2, 2)))


def make_data(times, causes):
    n = len(times)
    block = BlockData(BlockScheme('one', n, n, (0,)*n),
        tuple(Observation('one', t, c, 0) for t, c in zip(times, causes)))
    return BPCSExperiment((block,))


@pytest.fixture(scope='module')
def posterior_runs(prior):
    config = SamplingConfig(chains=4, warmup=500, draws=1200, seed=12345)
    data = load_example()
    return {(p, r): fit(data, prior='independent' if p == 'I' else prior,
                        restricted=r, config=config) for p in ('I', 'H') for r in (False, True)}


@pytest.mark.parametrize('seed', [4, 99, 721])
def test_joint_likelihood_and_hierarchical_normalizers(seed, prior):
    rng = np.random.default_rng(seed); data = load_example()
    a = rng.uniform(.4, 3); b = np.exp(rng.normal(size=(2, 2)))
    kap = np.exp(rng.normal(size=2)); lam = np.exp(rng.normal(size=2))
    pairs = dict(zip(('A', 'B'), map(tuple, b)))
    # Independently expanded likelihood: rows, counts, survival factors.
    expanded = log_censoring_constant(data)
    for i, block in enumerate(data.blocks):
        for row in block.observations:
            q = 1-row.time**a
            expanded += np.log(a)+np.log(b[i, row.cause-1])+(a-1)*np.log(row.time)
            expanded += ((1+row.withdrawals)*b[i].sum()-1)*np.log(q)
    assert expanded == pytest.approx(loglikelihood(data, a, pairs), abs=2e-13)
    independent = expanded-np.log(a)-np.log(b).sum()
    assert independent_logposterior(data, a, pairs) == pytest.approx(independent)
    expected = expanded+gamma.logpdf(a, prior.a.shape, scale=1/prior.a.rate)
    for s in range(2):
        expected += gamma.logpdf(kap[s], prior.kappa[s].shape, scale=1/prior.kappa[s].rate)
        expected += gamma.logpdf(lam[s], prior.lam[s].shape, scale=1/prior.lam[s].rate)
        expected += gamma.logpdf(b[:, s], kap[s], scale=1/lam[s]).sum()
    assert hierarchical_logposterior(data, a, pairs, kap, lam, prior) == pytest.approx(expected, abs=2e-12)


def test_marginal_integration_identity():
    data = load_example(); a = 1.4; b = np.array([[.4, .7], [.8, 1.2]])
    pairs = dict(zip(('A', 'B'), map(tuple, b)))
    cond = independent_conditionals(data, a)
    logconditional = 0.; gamma_constants = 0.
    for i, s in enumerate(('A', 'B')):
        d = cond[s]['cause_shapes']; W = np.exp(cond[s]['log_rate'])
        logconditional += gamma.logpdf(b[i], d, scale=1/W).sum()
        gamma_constants += gammaln(d).sum()
    actual = independent_logposterior(data, a, pairs)-logconditional
    expected = independent_log_marginal(data, a)+log_censoring_constant(data)+gamma_constants
    assert actual == pytest.approx(expected, abs=2e-13)
    # Numerical Gamma integrations, independent of the conjugate code.
    for d in (1, 2, 4):
        W = 1.7
        integral = quad(lambda x: x**(d-1)*np.exp(-W*x), 0, np.inf)[0]
        assert integral == pytest.approx(math.gamma(d)/W**d, rel=2e-10)


def test_all_hierarchical_conditionals_by_joint_density_ratios(prior):
    data = load_example(); a = 1.2
    b = np.array([[.3, .8], [.5, 1.1]]); kap = np.array([1.4, 2.1]); lam = np.array([.8, 1.3])
    pairs = dict(zip(('A', 'B'), map(tuple, b)))
    base = hierarchical_logposterior(data, a, pairs, kap, lam, prior)
    cond = hierarchical_conditionals(data, a, pairs, kap, lam, prior)
    for i in range(2):
        for s in range(2):
            other = b.copy(); other[i, s] *= 1.6
            diff = hierarchical_logposterior(data, a, dict(zip(('A', 'B'), map(tuple, other))), kap, lam, prior)-base
            shape, rate = cond['b_shape'][i, s], np.exp(cond['b_lograte'][i, s])
            oracle = gamma.logpdf(other[i, s], shape, scale=1/rate)-gamma.logpdf(b[i, s], shape, scale=1/rate)
            assert diff == pytest.approx(oracle, abs=1e-13)
    for s in range(2):
        other = lam.copy(); other[s] *= 1.7
        diff = hierarchical_logposterior(data, a, pairs, kap, other, prior)-base
        shape, rate = cond['lambda_shape'][s], cond['lambda_rate'][s]
        assert diff == pytest.approx(gamma.logpdf(other[s], shape, scale=1/rate)-gamma.logpdf(lam[s], shape, scale=1/rate))
        other = kap.copy(); other[s] *= 1.5
        diff = hierarchical_logposterior(data, a, pairs, other, lam, prior)-base
        oracle = kappa_logconditional(other[s], b[:, s], lam[s], prior.kappa[s])-kappa_logconditional(kap[s], b[:, s], lam[s], prior.kappa[s])
        assert diff == pytest.approx(oracle, abs=1e-13)
    other_a = 1.8
    assert hierarchical_logposterior(data, other_a, pairs, kap, lam, prior)-base == pytest.approx(
        a_logconditional(data, other_a, pairs, prior)-a_logconditional(data, a, pairs, prior), abs=1e-13)


@pytest.mark.parametrize('a,other_a', [(.01, .03), (.3, 1.7), (2., 5.), (25., 40.)])
def test_cached_a_kernel_matches_independent_row_likelihood(prior, a, other_a):
    from src.bayesian import _Context, _a_conditional_kernel
    data = load_example()
    pairs = {'A': (.3, .8), 'B': (.5, 1.1)}
    ctx = _Context(data)
    totals = np.array([sum(pairs[s]) for s in ctx.labels])
    fast = _a_conditional_kernel(ctx, other_a, totals, prior)-_a_conditional_kernel(ctx, a, totals, prior)
    oracle = a_logconditional(data, other_a, pairs, prior)-a_logconditional(data, a, pairs, prior)
    assert fast == pytest.approx(oracle, abs=2e-12)


@pytest.mark.parametrize('causes', [(1, 1), (2, 2), (1,)])
def test_independent_zero_counts_rejected_before_sampler(causes):
    data = make_data(np.linspace(.2, .7, len(causes)), causes)
    for restricted in (False, True):
        check = posterior_conditions(data, restricted=restricted)
        assert check['proper'] is not True
        with pytest.raises(PosteriorError):
            fit(data, prior='independent', restricted=restricted)


def test_independent_finite_and_infinite_moments():
    finite = make_data((.05, .5, .9), (1, 2, 1))
    infinite = make_data((.5, .6), (1, 2))
    for restricted in (False, True):
        c = posterior_conditions(finite, restricted=restricted)
        assert c['proper'] and c['moments']['b1[one]']['first'] == 'finite'
        c = posterior_conditions(infinite, restricted=restricted)
        assert c['proper'] and c['moments']['b1[one]']['first'] == 'infinite'
    # Equality D=y* also diverges: two times (.25,.5).
    equality = posterior_conditions(make_data((.25, .5), (1, 2)))
    assert equality['moments']['b1[one]']['first'] == 'infinite'


def test_hierarchical_propriety_with_zero_counts_and_moment_bounds(prior):
    data = make_data((.5,), (1,))
    c = posterior_conditions(data, prior=prior, restricted=True)
    assert c['proper'] and c['D'] == 0
    assert c['moments']['b1[one]']['first'] == 'finite'
    weak = replace(prior, a=GammaPrior(2, .01))
    c = posterior_conditions(data, prior=weak)
    assert c['proper'] and c['moments']['b1[one]']['first'] == 'unverified'
    assert all(c['moments'][s]['first'] == 'finite' for s in ('a', 'kappa1', 'lambda1'))


@pytest.mark.parametrize('a', [.3, 1., 4.])
def test_total_beta_jacobian_and_order_probability(a):
    data = load_example(); cond = independent_conditionals(data, a)
    for label, B, p in [('A', 1.3, .3), ('B', .7, .2)]:
        d1, d2 = cond[label]['cause_shapes']; W = np.exp(cond[label]['log_rate'])
        raw = gamma.pdf(B*p, d1, scale=1/W)*gamma.pdf(B*(1-p), d2, scale=1/W)*B
        transformed = gamma.pdf(B, d1+d2, scale=1/W)*beta.pdf(p, d1, d2)
        assert raw == pytest.approx(transformed, rel=2e-14)
        # Integrate conditional order probability directly against b2 density.
        probability = quad(lambda b2: gamma.cdf(b2, d1, scale=1/W)*gamma.pdf(b2, d2, scale=1/W), 0, np.inf)[0]
        assert probability == pytest.approx(betainc(d1, d2, .5), abs=1e-9)


@pytest.mark.parametrize('shape,rate,lower,upper', [(2., 1.3, 0., 2.), (3., .8, 2., np.inf),
    (2., 1., 1000., np.inf), (100., 1., 0., .001), (2., 1., 1000., 1000.00001)])
def test_truncated_gamma_probability_integral_transform(shape, rate, lower, upper):
    rng = np.random.default_rng(911)
    x = np.array([truncated_gamma(shape, rate, rng, lower=lower, upper=upper) for _ in range(700)])
    assert np.all((x > lower) & (x < upper))
    lo, hi = lower*rate, upper*rate
    # Oracle tail shapes in extreme cases reduce to shifted exponential/power.
    if lower == 1000.:
        # shape=2 has exact Q(x)=exp(-x)*(1+x).
        logs = -x+np.log1p(x); loglo = -lo+np.log1p(lo)
        loghi = -np.inf if np.isinf(hi) else -hi+np.log1p(hi)
        U = (-np.expm1(logs-loglo))/(-np.expm1(loghi-loglo))
    elif upper == .001:
        # At x<=.001, lower tail is near x**shape; exact ratios use stable
        # log-P series, independently compared to high precision below.
        U = np.exp(np.array([_log_gamma_tail(shape, v) for v in x])- _log_gamma_tail(shape, hi))
    else:
        U = (gamma.cdf(x*rate, shape)-gamma.cdf(lo, shape))/(gamma.cdf(hi, shape)-gamma.cdf(lo, shape))
    assert kstest(U, 'uniform').statistic < .065  # >99% DKW band at N=700
    assert abs(U.mean()-.5) < .045


def test_log_gamma_tail_against_independent_integral():
    # Integrate a scaled lower-tail density to avoid tiny absolute integrals.
    shape, upper = 100., .001
    scaled = quad(lambda y: y**(shape-1)*np.exp(upper*(1-y)), 0, 1, epsabs=1e-12)[0]
    expected = shape*np.log(upper)-upper-gammaln(shape)+np.log(scaled)
    assert _log_gamma_tail(shape, upper) == pytest.approx(expected, abs=2e-12)
    assert _log_gamma_tail(2., 1000., True) == pytest.approx(-1000+np.log(1001), abs=1e-12)


@pytest.mark.parametrize('seed', [5, 78])
def test_gamma_sampler_and_log_slice_invariant_target(seed):
    rng = np.random.default_rng(seed)
    x = np.array([_gamma_draw(2.5, np.log(1.7), rng) for _ in range(5000)])
    assert x.mean() == pytest.approx(2.5/1.7, abs=.065)
    assert x.var() == pytest.approx(2.5/1.7**2, abs=.08)
    current = 1.; retained = []
    # eta=log v; target Gamma(3,2) contributes Jacobian +eta.
    def target(z): return 3*z-2*np.exp(z)
    for j in range(4500):
        current = slice_step(target, current, rng)
        if j >= 500: retained.append(np.exp(current))
    assert np.mean(retained) == pytest.approx(1.5, abs=.06)
    assert np.var(retained) == pytest.approx(.75, abs=.09)


def test_slice_budget_failure_is_explicit():
    with pytest.raises(SamplingError):
        slice_step(lambda z: 0., 0., np.random.default_rng(1), max_steps=2)


def test_diagnostics_iid_stuck_and_autocorrelated():
    rng = np.random.default_rng(105)
    iid = rng.normal(size=(4, 3000)); good = chain_diagnostics(iid)
    assert good['rhat'] < 1.01 and good['ess_bulk'] > 8000
    assert good['mcse_mean'] == pytest.approx(1/np.sqrt(12000), rel=.3)
    shifted = iid+np.arange(4)[:, None]*2
    assert chain_diagnostics(shifted)['rhat'] > 1.3
    constant = chain_diagnostics(np.ones((4, 20)))
    assert np.isinf(constant['rhat']) and constant['ess_bulk'] == 0
    ar = np.zeros_like(iid)
    for i in range(1, ar.shape[1]): ar[:, i] = .8*ar[:, i-1]+iid[:, i]
    # AR(1) asymptotic ESS=N*(1-rho)/(1+rho).
    assert chain_diagnostics(ar)['ess_bulk'] == pytest.approx(12000/9, rel=.35)


@pytest.mark.parametrize('n,p', [(10, .95), (10, .8), (5, .01), (100, .95)])
def test_shortest_interval_indexing_and_mass(n, p):
    values = np.arange(n, dtype=float)
    ci = shortest_interval(values, p)
    assert ci.empirical_mass >= p
    assert ci.upper-ci.lower == math.ceil(n*p)-1
    assert ci.empirical_mass == pytest.approx(math.ceil(n*p)/n)


def test_interval_ties_and_multimodal_qualification():
    ci = shortest_interval([0, 0, 0, 1, 3], .5)
    assert ci.lower == ci.upper == 0 and ci.empirical_mass == .6
    assert 'contiguous' in ci.method and 'approximation' in ci.method


def test_four_chains_support_diagnostics_and_moments(posterior_runs):
    for (p, restricted), result in posterior_runs.items():
        assert result.status != 'numerical_failure', result.warnings
        assert result.converged, {k:v for k,v in result.diagnostics['parameters'].items() if v['rhat']>1.01 or min(v['ess_bulk'],v['ess_tail'])<100}
        assert result.draws.shape[:2] == (4, 1200)
        assert np.all(np.isfinite(result.draws)) and np.all(result.draws > 0)
        if restricted:
            for i in (1, 3): assert np.all(result.draws[:, :, i] <= result.draws[:, :, i+1])
        # No inadvertent order between facilities.
        assert np.any(result.draws[:, :, 1] > result.draws[:, :, 3])
        assert np.any(result.draws[:, :, 1] < result.draws[:, :, 3])
        assert all(v is not None for v in result.estimates.values())
        assert all(ci.empirical_mass >= .95 for ci in result.intervals.values())


def test_independent_shared_a_total_draws_and_truncated_beta(posterior_runs):
    u, r = posterior_runs['I', False], posterior_runs['I', True]
    np.testing.assert_array_equal(u.draws[:, :, 0], r.draws[:, :, 0])
    for j, d in [(1, (2, 1)), (3, (1, 3))]:
        U = u.draws[:, :, j]+u.draws[:, :, j+1]
        R = r.draws[:, :, j]+r.draws[:, :, j+1]
        # Two products and their sum reconstruct B with a few rounding steps.
        np.testing.assert_allclose(U, R, rtol=4*np.finfo(float).eps)
        p = r.draws[:, :, j]/R
        exactmean = d[0]/sum(d)*betainc(d[0]+1, d[1], .5)/betainc(*d, .5)
        assert p.mean() == pytest.approx(exactmean, abs=.012)
        assert kstest((betainc(*d, p)/betainc(*d, .5)).ravel(), 'uniform').statistic < .03


def marginal_quadrature_mean(data, coordinate=None):
    # Independent log-a quadrature, with the proved lower-tail asymptote.
    blocks = []
    T = sum(np.log(r.time) for b in data.blocks for r in b.observations)
    M = sum(b.scheme.m for b in data.blocks)
    for b in data.blocks:
        u = -np.log([r.time for r in b.observations]); w = 1+np.array(b.scheme.withdrawals)
        blocks.append((u, w))
    def kernel(z, moment=0):
        if z > 8: return 0.
        logweights = []
        if z < -30:
            logweights = [np.log(np.dot(w, -z-np.log(u))) for u,w in blocks]
            logf = -T-sum(np.log(u).sum()+len(u)*lw for (u,w),lw in zip(blocks,logweights))
        else:
            a = np.exp(z); logf = M*z+(a-1)*T
            for u, w in blocks:
                v = a*u; L = np.empty_like(v); near = v < np.log(2)
                L[near] = np.log(-np.expm1(-v[near]))
                L[~near] = np.log1p(-np.exp(-v[~near]))
                lw = np.log(-w@L); logweights.append(lw)
                logf -= L.sum()+len(u)*lw
        logmoment = moment*z
        if coordinate is not None and moment:
            i,c = coordinate
            logmoment = np.log(data.blocks[i].cause_counts[c+1])-logweights[i]
        return np.exp(logf+logmoment)
    norm = quad(lambda z: kernel(z), -np.inf, np.inf, epsabs=1e-8)[0]
    return quad(lambda z: kernel(z, 1), -np.inf, np.inf, epsabs=1e-8)[0]/norm


@pytest.mark.parametrize('coordinate',[None,(0,0),(0,1),(1,0),(1,1)])
def test_independent_posterior_against_one_dimensional_quadrature(posterior_runs,coordinate):
    r = posterior_runs['I', False]
    expected = marginal_quadrature_mean(load_example(),coordinate)
    name = 'a' if coordinate is None else f'b{coordinate[1]+1}[{"AB"[coordinate[0]]}]'
    mcse = r.diagnostics['parameters'][name]['mcse_mean']
    assert r.estimates[name] == pytest.approx(expected, abs=5*mcse)


def test_hierarchical_order_event_depends_on_a(prior):
    # Unequal cause rates destroy the shared-rate Beta probability.
    data = load_example(); probs = []
    for a in (.3, 4.):
        cond = hierarchical_conditionals(data, a, {'A':(.5,.6), 'B':(.4,.7)}, [1., 4.], [.1, 5.], prior)
        shapes = cond['b_shape'][0]; rates = np.exp(cond['b_lograte'][0])
        probs.append(betainc(*shapes, rates[0]/rates.sum()))
    assert abs(probs[0]-probs[1]) > .01


def test_reliability_draws_are_joint_function_averages(posterior_runs):
    r = posterior_runs['I', True]
    for label, j in [('A',1), ('B',3)]:
        a = r.draws[:, :, 0]; B = r.draws[:, :, j]+r.draws[:, :, j+1]
        p = r.draws[:, :, j]/B; S = (1-.5**a)**B
        entries = r.reliability[label]
        np.testing.assert_allclose(entries['survival@0.5']['draws'], S, rtol=2e-13)
        np.testing.assert_allclose(entries['hazard@0.5']['draws'], a*B*.5**(a-1)/(1-.5**a), rtol=2e-13)
        np.testing.assert_allclose(entries['median']['draws'], (1-2**(-1/B))**(1/a), rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose(entries['cause_1_incidence@0.5']['draws'], p*(1-S), atol=2e-15)
        assert entries['survival@0.5']['estimate'] == pytest.approx(S.mean())
        plugin = Kumaraswamy(a.mean(), B.mean()).sf(.5)
        assert abs(S.mean()-plugin) > .001


def test_unconverged_and_unverified_means_withheld(prior):
    cfg = SamplingConfig(chains=2, warmup=5, draws=16, seed=44)
    r = fit(load_example(), prior=prior, config=cfg)
    assert not r.converged and all(v is None for v in r.estimates.values())
    assert r.status == 'diagnostics_failed'
    # Proper but infinite b mean: still sample and return intervals, never a mean.
    data = make_data((.5,.6), (1,2))
    r = fit(data, prior='independent', config=replace(cfg, warmup=100, draws=100, seed=99))
    assert r.status != 'numerical_failure', r.warnings
    assert r.moments['b1[one]']['first'] == 'infinite'
    assert r.estimates['b1[one]'] is None
    assert r.diagnostics['parameters']['b1[one]']['mcse_mean'] is None


def test_reproducibility_and_explicit_numerical_failure():
    cfg = SamplingConfig(chains=2, warmup=10, draws=20, seed=8)
    a = fit(load_example(), prior='independent', config=cfg)
    b = fit(load_example(), prior='independent', config=cfg)
    np.testing.assert_array_equal(a.draws,b.draws)
    failed = fit(load_example(), prior='independent', config=replace(cfg, slice_width=1e-9,max_slice_steps=1))
    assert failed.status == 'numerical_failure' and not failed.converged
    assert failed.draws.size == 0 and failed.estimates == {}


@pytest.mark.parametrize('key,value', [('chains',1),('draws',7),('warmup',-1),('seed',True),
    ('max_slice_steps',0),('slice_width',0),('max_rhat',1),('min_ess',np.inf),('probability',1)])
def test_invalid_sampler_configuration(key,value):
    with pytest.raises(ValueError): SamplingConfig(**{key:value})


@pytest.mark.parametrize('value', [0, -1, True, np.nan, np.inf])
def test_invalid_gamma_prior(value):
    with pytest.raises(ValueError): GammaPrior(value, 1)
    with pytest.raises(ValueError): GammaPrior(1, value)


def test_invalid_models_and_intervals(prior):
    with pytest.raises(ValueError): HierarchicalPrior(GammaPrior(1,1),(GammaPrior(1,1),),prior.lam)
    with pytest.raises(ValueError): fit(load_example(),prior='unknown')
    with pytest.raises(ValueError): fit(load_example(),prior=prior,restricted='across')
    with pytest.raises(ValueError): fit(load_example(),prior=prior,times=[0,1])
    with pytest.raises(ValueError): shortest_interval([1,np.nan])
    with pytest.raises(ValueError): chain_diagnostics(np.ones((1,20)))
    with pytest.raises(ValueError): truncated_gamma(2,1,np.random.default_rng(1),lower=2,upper=1)


def test_propriety_gap_resolves_adjacent_distinct_times():
    times = (.5, np.nextafter(.5, 1.))
    data = make_data(times, (1, 2))
    check = posterior_conditions(data)
    assert check['proper'] and check['D'] > 0
    assert check['D'] == pytest.approx(np.log1p((times[1]-times[0])/times[0]), rel=1e-15)


@pytest.mark.parametrize('component', ['a', 'kappa'])
def test_nonstandard_conditional_slice_against_quadrature(component, prior):
    data = load_example(); b = {'A':(.3,.8), 'B':(.5,1.1)}
    raw = (lambda v: a_logconditional(data, v, b, prior)) if component == 'a' else (
        lambda v: kappa_logconditional(v, [.3,.5], .8, prior.kappa[0]))
    offset = raw(1.)
    def integrand(v, moment=0):
        if v <= 0: return 0.
        return np.exp(raw(v)-offset)*v**moment
    norm = quad(integrand, 0, np.inf, epsabs=1e-9)[0]
    expected = quad(lambda v: integrand(v,1), 0, np.inf, epsabs=1e-9)[0]/norm
    rng = np.random.default_rng(901); eta = 0.; x = []
    def target(z): return raw(np.exp(z))+z
    for j in range(3400):
        eta = slice_step(target, eta, rng)
        if j >= 400: x.append(np.exp(eta))
    diag = chain_diagnostics(np.array(x).reshape(2,-1))
    assert np.mean(x) == pytest.approx(expected, abs=5*diag['mcse_mean'])


@pytest.mark.parametrize('restricted', [False,True])
def test_hierarchical_zero_counts_produce_positive_ordered_draws(prior, restricted):
    data = make_data((.1,.4,.8), (2,2,2))
    cfg = SamplingConfig(chains=2,warmup=100,draws=100,seed=18)
    r = fit(data,prior=prior,restricted=restricted,config=cfg)
    assert r.draws.size and r.status != 'numerical_failure', r.warnings
    assert np.all(r.draws>0)
    if restricted:
        assert np.all(r.draws[:,:,1]<=r.draws[:,:,2])


def test_independent_second_seed_matches_quadrature():
    cfg = SamplingConfig(chains=4,warmup=500,draws=1200,seed=816)
    r = fit(load_example(),prior='independent',config=cfg)
    assert r.converged
    expected = marginal_quadrature_mean(load_example())
    assert r.estimates['a'] == pytest.approx(expected,abs=5*r.diagnostics['parameters']['a']['mcse_mean'])


def test_folded_rank_diagnostic_detects_scale_mismatch():
    rng = np.random.default_rng(316)
    chains = rng.normal(size=(4,2000))*np.array([.1,1,2,8])[:,None]
    assert chain_diagnostics(chains)['rhat'] > 1.2


def test_large_finite_sample_diagnostics_do_not_overflow():
    rng = np.random.default_rng(4)
    draws = rng.uniform(.1,.9,size=(4,500))*1e308
    with np.errstate(over='raise', invalid='raise'):
        diag = chain_diagnostics(draws)
    assert np.isfinite(diag['mcse_mean']) and diag['rhat'] < 1.02


def test_mission_time_keys_preserve_precision(posterior_runs):
    r = posterior_runs['I',False]
    t = [.123456789123,.123456789124]
    values = posterior_reliability(r.draws[:,:8],r.names,('A','B'),t)
    assert 'survival@0.123456789123' in values['A']
    assert 'survival@0.123456789124' in values['A']


def test_reliability_numerical_failure_is_structured(monkeypatch):
    # Inject an unrepresentable mission-time hazard, independently of MCMC.
    monkeypatch.setattr(Kumaraswamy,'hazard',lambda self,t: np.inf)
    r = fit(load_example(),prior='independent',
            config=SamplingConfig(chains=2,warmup=10,draws=20,seed=8),
            times=(np.nextafter(0.,1.),))
    assert r.status == 'numerical_failure' and not r.converged
    assert all(v is None for v in r.estimates.values())


@pytest.mark.parametrize('prior_family', ['I','H'])
def test_three_block_parameter_order_and_support(prior,prior_family):
    blocks = []
    template = load_example().blocks[0]
    for name in ('z','a','third'):
        scheme = replace(template.scheme,block=name)
        blocks.append(BlockData(scheme,tuple(replace(row,block=name) for row in template.observations)))
    r = fit(BPCSExperiment(tuple(blocks)),prior='independent' if prior_family=='I' else prior,
            restricted=True, config=SamplingConfig(chains=2,warmup=50,draws=50,seed=97))
    assert r.status != 'numerical_failure', r.warnings
    assert r.names[:7] == ('a','b1[z]','b2[z]','b1[a]','b2[a]','b1[third]','b2[third]')
    assert all(np.all(r.draws[:,:,j]<=r.draws[:,:,j+1]) for j in (1,3,5))


def test_extreme_valid_hyperprior_reports_initialization_range_failure(prior):
    extreme = replace(prior, a=GammaPrior(1e308,1e-320))
    r = fit(load_example(),prior=extreme,
            config=SamplingConfig(chains=2,warmup=0,draws=8,seed=9))
    assert r.status == 'numerical_failure' and not r.converged
    assert 'initialization' in r.diagnostics['error']


def test_rare_independent_order_event_uses_beta_inversion_not_rejection():
    n = 1001
    data = make_data(np.linspace(.1,.9,n),(1,)*(n-1)+(2,))
    r = fit(data,prior='independent',restricted=True,
            config=SamplingConfig(chains=2,warmup=10,draws=20,seed=33))
    assert r.status != 'numerical_failure', r.warnings
    B = r.draws[:,:,1]+r.draws[:,:,2]; p = r.draws[:,:,1]/B
    assert np.all((p>.49)&(p<=.5))
    assert betainc(1000,1,.5) < 1e-300


@pytest.mark.parametrize('restricted',[False,True])
def test_hierarchical_reproducibility(prior,restricted):
    cfg = SamplingConfig(chains=2,warmup=10,draws=20,seed=411)
    x=fit(load_example(),prior=prior,restricted=restricted,config=cfg)
    y=fit(load_example(),prior=prior,restricted=restricted,config=cfg)
    assert x.status != 'numerical_failure', x.warnings
    np.testing.assert_array_equal(x.draws,y.draws)


def test_hazard_moments_are_not_inferred_from_total_mean():
    data=make_data((.5,.6),(1,2))
    r=fit(data,prior='independent',times=(.5,.99),
          config=SamplingConfig(chains=2,warmup=100,draws=100,seed=99))
    assert r.status != 'numerical_failure',r.warnings
    assert r.moments['b1[one]']['first']=='infinite'
    assert r.reliability['one']['hazard@0.5']['moments']['second']=='finite'
    assert r.reliability['one']['hazard@0.99']['moments']['first']=='infinite'
    assert r.reliability['one']['hazard@0.99']['estimate'] is None


def test_finite_first_infinite_second_moment_withholds_mcse():
    data=make_data((.2,.5),(1,2))
    r=fit(data,prior='independent',
          config=SamplingConfig(chains=2,warmup=100,draws=100,seed=99))
    assert r.status != 'numerical_failure',r.warnings
    assert r.moments['b1[one]']==dict(first='finite',second='infinite')
    assert r.diagnostics['parameters']['b1[one]']['mcse_mean'] is None
