"""Independent deterministic checks of classical inference formulas."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import binom, norm

from experiments.stage2_example import load_example
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.likelihood import loglikelihood
from src.mle import fit_mle
from src.model import Kumaraswamy
from src.order_restricted import fit_order_restricted_mle
from src.uncertainty import (InformationError, binomial_interval,
    covariance_from_information, observed_information, parameter_inference,
    reliability_estimates, reliability_gradients, reliability_inference)


def data_with_causes(causes):
    return BPCSExperiment(tuple(BlockData(BlockScheme(str(i), len(cs), len(cs), (0,)*len(cs)),
        tuple(Observation(str(i), float(t), c, 0) for t, c in zip(np.linspace(.12, .82, len(cs)), cs)))
        for i, cs in enumerate(causes)))


def numerical_hessian(f, x, step=2e-4):
    x = np.asarray(x, dtype=float)
    out = np.empty((len(x), len(x)))
    for i in range(len(x)):
        ei = np.eye(len(x))[i]*step
        for j in range(len(x)):
            ej = np.eye(len(x))[j]*step
            out[i, j] = -(f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*step**2)
    return out


@pytest.mark.parametrize('k', [1, 2, 3])
@pytest.mark.parametrize('coordinates', ['total', 'original'])
def test_hessian_against_independent_likelihood(k, coordinates):
    data = data_with_causes([(1, 2, 1, 2)]*k)
    labels = [b.scheme.block for b in data.blocks]
    if coordinates == 'total':
        x = np.array([1.3]+[1.1+.1*i for i in range(k)]+[.35+.05*i for i in range(k)])
        def unpack(v):
            return {s: (v[1+i]*v[1+k+i], v[1+i]*(1-v[1+k+i])) for i, s in enumerate(labels)}
    else:
        x = np.array([1.3]+[v for i in range(k) for v in (.4+.1*i, .7+.1*i)])
        def unpack(v):
            return {s: tuple(v[1+2*i:3+2*i]) for i, s in enumerate(labels)}
    fun = lambda v: loglikelihood(data, v[0], unpack(v))
    J = observed_information(data, x[0], unpack(x), coordinates=coordinates)
    np.testing.assert_allclose(J.matrix, numerical_hessian(fun, x), rtol=2e-5, atol=2e-6)
    assert J.matrix.shape == (1+2*k, 1+2*k)
    np.testing.assert_array_equal(J.matrix, J.matrix.T)


def test_nonlinear_hessian_transformation_requires_score_correction():
    data = load_example()
    a, Bs, ps = 1.3, [1.1, .9], [.3, .4]
    b = {s: (B*p, B*(1-p)) for s, B, p in zip(('A', 'B'), Bs, ps)}
    Jq = observed_information(data, a, b).matrix
    Jo = observed_information(data, a, b, coordinates='original').matrix
    A = np.zeros((5, 5)); A[0, 0] = 1
    correction = np.zeros((5, 5))
    for i, (block, B, p) in enumerate(zip(data.blocks, Bs, ps)):
        j = 1+2*i
        A[1+i, j:j+2] = 1
        A[3+i, j:j+2] = [(1-p)/B, -p/B]
        d = block.cause_counts
        Up = d[1]/p-d[2]/(1-p)
        correction[j:j+2, j:j+2] = Up*np.array([[-2*(1-p), 2*p-1], [2*p-1, 2*p]])/B**2
    naive = A.T@Jq@A
    assert np.max(abs(naive-Jo)) > 1
    np.testing.assert_allclose(naive-correction, Jo, atol=1e-13)


def test_covariance_transformation_and_se():
    data = load_example(); fit = fit_mle(data)
    report = parameter_inference(data, fit)
    J = observed_information(data, fit.a, fit.b_by_block, coordinates='original').matrix
    np.testing.assert_allclose(report.covariance_original, np.linalg.solve(J, np.eye(5)), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(report.time_covariance, report.covariance_total[:3, :3], rtol=1e-13)
    np.testing.assert_array_equal(report.covariance_total[:3, 3:], 0)
    for name, variance in zip(report.original_names, np.diag(report.covariance_original)):
        assert report.standard_errors[name] == pytest.approx(np.sqrt(variance))
    for name, value in report.estimates.items():
        se = report.standard_errors[name]
        ci = report.intervals[name]['wald']
        assert ci.lower == pytest.approx(value-norm.ppf(.975)*se)
        if not name.startswith('p['):
            ci = report.intervals[name]['log']
            assert ci.lower == pytest.approx(value*np.exp(-norm.ppf(.975)*se/value))
            assert 0 < ci.lower < value < ci.upper
        else:
            ci = report.intervals[name]['logit']
            assert ci.upper == pytest.approx(expit(logit(value)+norm.ppf(.975)*se/value/(1-value)))
            assert 0 < ci.lower < value < ci.upper < 1


@pytest.mark.parametrize('matrix', [np.zeros((2, 2)), [[1, 2], [2, 1]], [[1, 1], [1, 1]],
    [[1, 0], [1, 1]], [[1, np.nan], [np.nan, 1]], [1, 2], np.empty((0, 0))])
def test_invalid_information(matrix):
    with pytest.raises(InformationError):
        covariance_from_information(matrix)


def test_conditioning_equilibrated_and_no_regularization():
    np.testing.assert_allclose(covariance_from_information(np.diag([1e-20, 1e20])), np.diag([1e20, 1e-20]))
    with pytest.raises(InformationError, match='ill-conditioned'):
        covariance_from_information([[1, 1-1e-10], [1-1e-10, 1]], max_condition=1e8)
    report = parameter_inference(load_example(), fit_mle(load_example()), max_condition=1)
    assert report.time_covariance is None and report.covariance_total is None
    assert all(x is None for x in report.standard_errors.values())
    assert report.intervals['p[A]']['exact'].status == 'available'


@pytest.mark.parametrize('m', [1, 7, 20])
def test_binomial_extreme_reference_values(m):
    assert binomial_interval(0, m).upper == pytest.approx(1-.025**(1/m))
    assert binomial_interval(m, m).lower == pytest.approx(.025**(1/m))


@pytest.mark.parametrize('p', [.001, .05, .25, .5])
@pytest.mark.parametrize('m', [3, 10, 40])
def test_restricted_exact_set_coverage_by_enumeration(p, m):
    coverage = 0
    for d in range(m+1):
        ci = binomial_interval(d, m, restricted=True)
        if ci.status != 'empty' and ci.lower <= p <= ci.upper:
            coverage += binom.pmf(d, m, p)
    assert coverage >= .95-1e-12


def test_empty_set_is_preserved():
    ci = binomial_interval(10, 10, restricted=True)
    assert ci.status == 'empty' and ci.lower is None and ci.upper is None


def test_restricted_inference_and_invariant_time_uncertainty():
    data = load_example(); u = fit_mle(data); r = fit_order_restricted_mle(data)
    U = parameter_inference(data, u); R = parameter_inference(data, r)
    assert R.diagnostics['probability_states'] == {'A': 'active_equality', 'B': 'inactive'}
    np.testing.assert_allclose(U.time_covariance, R.time_covariance, atol=1e-14)
    assert R.covariance_original is None and R.covariance_total is None
    for s in ('A', 'B'):
        assert R.standard_errors[f'b1[{s}]'] is None
        assert R.standard_errors[f'p[{s}]'] is None
        ci = R.intervals[f'p[{s}]']['exact']
        assert ci.upper <= .5
    for t in (.2, .5, .9):
        ur = reliability_inference(data, u, t); rr = reliability_inference(data, r, t)
        for s in ('A', 'B'):
            for name in ('survival', 'hazard', 'median'):
                for key in ('estimate', 'standard_error'):
                    assert ur[s][name][key] == pytest.approx(rr[s][name][key], rel=1e-14)
            assert rr[s]['cause_1_incidence']['standard_error'] is None
            assert rr[s]['cause_1_incidence']['interval'] is None
            assert rr[s]['cause_2_probability']['interval'].lower >= .5


@pytest.mark.parametrize('causes', [(2, 2, 2, 2), (1, 1, 1, 1), (1, 2, 1, 2)])
def test_zero_boundaries_and_equal_counts(causes):
    data = data_with_causes([causes]); u = fit_mle(data); r = fit_order_restricted_mle(data)
    U = parameter_inference(data, u); R = parameter_inference(data, r)
    assert U.time_covariance is not None and R.time_covariance is not None
    if len(set(causes)) == 1:
        assert U.covariance_original is None
        assert U.standard_errors['p[0]'] is None
        with pytest.raises(InformationError):
            observed_information(data, u.a, u.b_by_block)
    state = R.diagnostics['probability_states']['0']
    assert state == ('zero_boundary' if causes[0] == 2 and len(set(causes)) == 1 else 'active_equality')
    values = reliability_estimates(data, u, .5)['0']
    assert values['cause_1_incidence']+values['cause_2_incidence'] == pytest.approx(1-values['survival'])
    assert all(np.isfinite(v) for v in values.values())


def point_values(a, B, p, t):
    model = Kumaraswamy(a, B)
    return dict(survival=model.sf(t), hazard=model.hazard(t), median=model.median(),
        cause_1_probability=p, cause_2_probability=1-p,
        cause_1_incidence=p*model.cdf(t), cause_2_incidence=(1-p)*model.cdf(t))


@pytest.mark.parametrize('a,B,p,t', [(1.3, .8, .3, .4), (.25, 4., .7, .9), (4., .3, .5, .1)])
def test_reliability_gradients_finite_difference(a, B, p, t):
    x = np.array([a, B, p]); analytic = reliability_gradients(*x, t)
    step = 1e-6
    for j in range(3):
        e = np.eye(3)[j]*step
        plus, minus = point_values(*(x+e), t), point_values(*(x-e), t)
        for name in analytic:
            assert analytic[name][j] == pytest.approx((plus[name]-minus[name])/(2*step), rel=3e-5, abs=1e-9)


def test_delta_reference_quadratic_form_and_intervals():
    data = load_example(); fit = fit_mle(data)
    report = parameter_inference(data, fit)
    values = reliability_inference(data, fit, .5)
    for i, s in enumerate(('A', 'B')):
        b1, b2 = fit.b_by_block[s]; B = b1+b2; p = b1/B
        gradients = reliability_gradients(fit.a, B, p, .5)
        for name, g in gradients.items():
            full = np.zeros(5); full[[0, 1+i, 3+i]] = g
            expected = np.sqrt(full@report.covariance_total@full)
            assert values[s][name]['standard_error'] == pytest.approx(expected)
            ci = values[s][name]['interval']
            if name == 'hazard':
                assert 0 < ci.lower < ci.upper
            else:
                assert 0 <= ci.lower < ci.upper <= 1
    assert values['A']['survival']['estimate'] == pytest.approx((1-.5**fit.a)**sum(fit.b_by_block['A']))


@pytest.mark.parametrize('bad', [0, 1, -1, float('nan'), True])
def test_invalid_confidence(bad):
    with pytest.raises(ValueError):
        binomial_interval(1, 3, confidence=bad)


@pytest.mark.parametrize('d,m', [(-1, 2), (3, 2), (True, 2), (1., 2), (0, 0)])
def test_invalid_counts(d, m):
    with pytest.raises(ValueError):
        binomial_interval(d, m)


def test_invalid_fits_and_parameters():
    data = load_example(); fit = fit_mle(data)
    with pytest.raises(ValueError):
        parameter_inference(data, replace(fit, converged=False))
    with pytest.raises(ValueError):
        parameter_inference(data, replace(fit, a=fit.a*2))
    with pytest.raises(ValueError):
        parameter_inference(data, fit_order_restricted_mle(data), restricted=False)
    with pytest.raises(ValueError):
        observed_information(data, -1, fit.b_by_block)
    with pytest.raises(ValueError):
        observed_information(data, fit.a, {'Z': (1, 2)})
    with pytest.raises(ValueError):
        observed_information(data, fit.a, fit.b_by_block, coordinates='unknown')
    with pytest.raises(ValueError):
        observed_information(data, fit.a, fit.b_by_block, coordinates='original', time_only=True)
    for t in (0, 1, -1, float('inf')):
        with pytest.raises(ValueError):
            reliability_inference(data, fit, t)
    with pytest.raises(ValueError):
        reliability_gradients(1, 2, 1.1, .5)


def test_information_small_shape_limits_are_stable():
    data = load_example(); a = 1e-9; bs = {'A': (.4, .6), 'B': (.3, .5)}
    J = observed_information(data, a, bs, time_only=True).matrix
    expected = sum(sum(bs[b.scheme.block])*sum(1+r for r in b.scheme.withdrawals) for b in data.blocks)
    assert J[0, 0]*a*a == pytest.approx(expected, rel=1e-8)
    for i, block in enumerate(data.blocks):
        assert J[0, 1+i]*a == pytest.approx(-sum(1+r for r in block.scheme.withdrawals), rel=1e-8)


def test_empty_set_propagation_and_zero_cause_reliability():
    data = data_with_causes([(1,)*10]); fit = fit_order_restricted_mle(data)
    report = parameter_inference(data, fit)
    assert report.intervals['p[0]']['exact'].status == 'empty'
    for name in ('cause_1_probability', 'cause_2_probability'):
        assert reliability_inference(data, fit, .5)['0'][name]['interval'].status == 'empty'
    data = data_with_causes([(2,)*5]); fit = fit_mle(data)
    entries = reliability_inference(data, fit, .5)['0']
    assert entries['survival']['standard_error'] is not None
    assert entries['cause_1_incidence']['estimate'] == 0
    assert entries['cause_1_incidence']['standard_error'] is None


def test_binomial_information_known_variance_and_fit_methods():
    from src.mle import fit_direct
    data = load_example(); fit = fit_direct(data)
    report = parameter_inference(data, fit)
    for i, block in enumerate(data.blocks):
        p = report.estimates[f'p[{block.scheme.block}]']
        assert report.covariance_total[3+i, 3+i] == pytest.approx(p*(1-p)/block.scheme.m, rel=1e-6)
    with pytest.raises(ValueError):
        parameter_inference(data, replace(fit, loglikelihood=float('inf')))
    with pytest.raises(ValueError):
        observed_information(data, fit.a, {'A': (True, 1), 'B': (1, 1)})


def test_interval_numerical_failure_explicit_without_clipping():
    from src.uncertainty import _wald
    assert _wald(1., 1e4, .95, 'log').status == 'unavailable'
    for value in (0., 1.):
        assert _wald(value, .1, .95, 'logit').status == 'unavailable'
