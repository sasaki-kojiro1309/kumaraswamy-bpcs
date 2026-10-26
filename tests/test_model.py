"""Independent integral, calculus, and competing-risk checks for Stage 1."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import beta

from src.model import CompetingRisks, Kumaraswamy


@pytest.mark.parametrize("a,b", [(1, 1), (2, 3), (.6, 1.7), (1.8, .7), (.7, .8)])
def test_density_integrates_and_transformed_beta(a, b):
    model = Kumaraswamy(a, b)
    integral, error = quad(model.pdf, 0, 1, epsabs=1e-9)
    assert integral == pytest.approx(1, abs=2e-8)
    assert error < 2e-8
    # Independent change-of-variable oracle: T**a ~ Beta(1,b).
    t = np.linspace(.01, .99, 31)
    assert_allclose(model.cdf(t), beta.cdf(t**a, 1, b), atol=2e-15)
    assert_allclose(model.pdf(t), beta.pdf(t**a, 1, b) * a*t**(a-1))


@pytest.mark.parametrize("a,b", [(1, 1), (2.3, .8), (.6, 3)])
def test_calculus_survival_hazard_and_median(a, b):
    model = Kumaraswamy(a, b)
    t = np.linspace(.05, .95, 20)
    eps = 1e-6
    assert_allclose(model.cdf(t) + model.sf(t), 1, atol=2e-16)
    assert_allclose((model.cdf(t+eps)-model.cdf(t-eps))/(2*eps), model.pdf(t), rtol=2e-8)
    assert_allclose(-(model.logsf(t+eps)-model.logsf(t-eps))/(2*eps), model.hazard(t), rtol=2e-8)
    assert_allclose(model.hazard(t), model.pdf(t)/model.sf(t))
    assert np.all(np.diff(model.cdf(t)) > 0)
    assert model.sf(model.median()) == pytest.approx(.5)


@pytest.mark.parametrize("a,b1,b2", [(1, 1, 1), (2, .8, 1.4), (.7, 1.3, .5)])
def test_competing_risks_from_independent_latent_lifetimes(a, b1, b2):
    model = CompetingRisks(a, b1, b2)
    t = np.linspace(.01, .99, 50)
    first, second = Kumaraswamy(a, b1), Kumaraswamy(a, b2)
    assert_allclose(model.minimum.sf(t), first.sf(t)*second.sf(t))
    assert_allclose(model.cause_pdf(t, 1), first.pdf(t)*second.sf(t))
    assert_allclose(model.cause_pdf(t, 2), second.pdf(t)*first.sf(t))
    assert_allclose(model.cause_pdf(t, 1)+model.cause_pdf(t, 2), model.minimum.pdf(t))
    assert_allclose(model.minimum.hazard(t), first.hazard(t)+second.hazard(t))
    assert_allclose(model.cause_cdf(t, 1)+model.cause_cdf(t, 2), model.minimum.cdf(t))
    for cause in (1, 2):
        mass = quad(lambda x: model.cause_pdf(x, cause), 0, 1)[0]
        assert mass == pytest.approx(model.cause_probability(cause), abs=2e-8)
    assert model.minimum.sf(model.minimum.median()) == pytest.approx(.5)


def test_known_uniform_case_and_support_conventions():
    model = Kumaraswamy(1, 1)
    t = np.array([-np.inf, -1, 0, .25, 1, 2, np.inf])
    assert_allclose(model.cdf(t), [0, 0, 0, .25, 1, 1, 1])
    assert_allclose(model.pdf(t), [0, 0, 0, 1, 0, 0, 0])
    assert model.hazard(.25) == pytest.approx(1/.75)
    assert model.median() == pytest.approx(.5)
    assert isinstance(model.pdf(.25), float)
    assert model.cdf([[.2, .3], [.4, .5]]).shape == (2, 2)
    assert model.cdf([]).shape == (0,)


@pytest.mark.parametrize("bad", [0, -1, np.inf, -np.inf, np.nan, [1, 2]])
def test_invalid_parameters(bad):
    for args in [(bad, 1), (1, bad)]:
        with pytest.raises(ValueError):
            Kumaraswamy(*args)
    for args in [(bad, 1, 2), (1, bad, 2), (1, 2, bad)]:
        with pytest.raises(ValueError):
            CompetingRisks(*args)


def test_invalid_times_causes_and_combined_shape():
    model = Kumaraswamy(2, 3)
    for method in [model.pdf, model.logpdf, model.cdf, model.sf, model.logsf, model.hazard]:
        with pytest.raises(ValueError):
            method([.2, np.nan])
    for t in [0, 1, -.1, 1.1, np.inf]:
        with pytest.raises(ValueError):
            model.hazard(t)
    competing = CompetingRisks(2, 3, 4)
    for cause in [0, 3, None, True, "1"]:
        with pytest.raises(ValueError):
            competing.cause_pdf(.5, cause)
    with pytest.raises(ValueError):
        CompetingRisks(1, 1e308, 1e308)


def test_tail_stability_without_clipping():
    tiny = 1e-100
    model = Kumaraswamy(2, 3)
    # Naive 1-(1-t**a)**b rounds to zero here; expm1 retains the mass.
    # Log/exp roundoff scales with |log(tiny)|; allow 1e-13 relative error.
    assert model.cdf(tiny) == pytest.approx(3e-200, rel=1e-13, abs=0)
    t = np.nextafter(1., 0.)
    small_a = Kumaraswamy(1e-3, 2)
    logq = np.log(-np.expm1(1e-3*np.log(t)))
    assert small_a.logsf(t) == pytest.approx(2*logq)
    assert np.isfinite(small_a.logpdf(t))
    assert small_a.sf(t) > 0
    heavy = Kumaraswamy(2, 1000)
    assert heavy.sf(.99) == 0  # ordinary probability underflows in float64
    assert np.isfinite(heavy.logsf(.99))
    assert np.isfinite(heavy.hazard(.99))  # evaluated without 0/0
    risks = CompetingRisks(2, 400, 600)
    assert np.isfinite(risks.cause_logpdf(.99, 1))


@pytest.mark.parametrize('a', [1e-300, 1e-310, 1e-320])
def test_subnormal_power_keeps_finite_log_density_and_hazard(a):
    # For tiny a, q=1-t**a ~ a*(-log t). This is a log-scale limit,
    # not a replacement of the parameter or time by epsilon.
    t = np.nextafter(1., 0.)
    model = Kumaraswamy(a, 1.)
    with np.errstate(all='raise'):
        expected_logq = np.log(a)+np.log(-np.log(t))
        assert model.logsf(t) == pytest.approx(expected_logq, abs=2e-13)
        assert model.logpdf(t) == pytest.approx(np.log(a)-np.log(t), abs=2e-13)
        expected_hazard = 1/(t*(-np.log(t)))
        assert model.hazard(t) == pytest.approx(expected_hazard, rel=2e-13)


@pytest.mark.parametrize('a', [460., 465., 500.])
def test_large_shape_rescues_representable_mass_after_power_underflow(a):
    model = Kumaraswamy(a, 1e308)
    # t**a underflows, but b*t**a is representable. Higher terms in the
    # expansion of log(1-t**a) are negligible in the subnormal-power tail.
    expected = np.exp(np.log(1e308)+a*np.log(.2))
    assert model.cdf(.2) == pytest.approx(-np.expm1(-expected), rel=2e-13, abs=0)
    assert model.logsf(.2) == pytest.approx(-expected, rel=2e-13, abs=0)
