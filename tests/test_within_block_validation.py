"""Additional invariance, KKT, independent SLSQP, and float64 stress checks."""

from math import log

import numpy as np
import pytest
from numpy.testing import assert_allclose

from experiments.stage2_example import load_example
from experiments.within_block_validation import _JointObjective, fit_slsqp, validation_report
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.mle import conditional_estimates, fit_mle, profile_loglikelihood, score
from src.model import Kumaraswamy
from src.order_restricted import fit_order_restricted_mle


def make_block(causes, times, label="A"):
    r = (1,)+(0,)*(len(times)-2)+(2,) if len(times) > 1 else (2,)
    return BlockData(BlockScheme(label, len(times)+sum(r), len(times), r),
                     tuple(Observation(label, t, c, removal)
                           for t, c, removal in zip(times, causes, r)))


@pytest.mark.parametrize("times", [(.01, .04, .1, .2, .4, .7), (.1, .2, .3, .4, .6, .8)])
def test_already_equal_unrestricted_pair_is_retained_exactly(times):
    complete = BlockData(BlockScheme('A', 6, 6, (0,)*6), tuple(
        Observation('A', t, 1 if j % 2 == 0 else 2, 0) for j, t in enumerate(times)))
    data = BPCSExperiment((complete,))
    unrestricted, restricted = fit_mle(data), fit_order_restricted_mle(data)
    assert restricted.converged
    assert restricted.b_by_block == unrestricted.b_by_block
    assert restricted.loglikelihood == unrestricted.loglikelihood
    assert restricted.diagnostics['active_constraints']['A']
    assert not restricted.diagnostics['pooled_blocks']['A']


def test_positive_active_equality_satisfies_raw_kkt_not_zero_cause_scores():
    data = load_example()
    r = fit_order_restricted_mle(data)
    raw = score(data, r.a, r.b_by_block)
    B = sum(r.b_by_block['A'])
    multiplier = (2-1)/B
    assert raw['a'] == pytest.approx(0, abs=1e-10)
    assert raw['b_by_block']['A'] == pytest.approx((multiplier, -multiplier), abs=1e-10)
    assert raw['b_by_block']['B'] == pytest.approx((0, 0), abs=1e-10)


@pytest.mark.parametrize("causes", [(1, 2, 2), (1, 1, 2), (1, 1, 1), (2, 2, 2)])
def test_full_minimum_distribution_and_block_total_are_preserved(causes):
    data = BPCSExperiment((make_block(causes, (.15, .4, .7)),))
    u, r = fit_mle(data), fit_order_restricted_mle(data)
    assert r.converged and r.a == u.a
    Bu, Br = sum(u.b_by_block['A']), sum(r.b_by_block['A'])
    assert Br == pytest.approx(Bu, rel=1e-15)
    grid = np.linspace(.001, .999, 100)
    assert_allclose(Kumaraswamy(r.a, Br).sf(grid), Kumaraswamy(u.a, Bu).sf(grid),
                    rtol=1e-13, atol=0)
    assert abs(r.diagnostics['relative_total_preservation_residuals']['A']) < 1e-14
    assert r.b_by_block['A'][0]/Br == pytest.approx(min(causes.count(1)/len(causes), .5))


def test_restricted_profile_penalty_is_constant_even_with_both_zero_count_directions():
    cases = [(1, 1, 2), (1, 2, 2), (1, 1, 1), (2, 2, 2)]
    data = BPCSExperiment(tuple(make_block(c, (.15, .4, .7), str(i)) for i, c in enumerate(cases)))
    r = fit_order_restricted_mle(data)
    objective = _JointObjective(data)
    for a in (.02, .3, 1.7, 10., 100.):
        conditional = conditional_estimates(data, a)
        v = [log(a/objective.a_scale)]
        for block in data.blocks:
            B = sum(conditional[block.scheme.block])
            p = min(block.cause_counts[1]/block.scheme.m, .5)
            v.extend([log(B), p])
        loss, _ = objective(np.array(v))
        assert profile_loglikelihood(data, a)+loss == pytest.approx(
            r.diagnostics['loglikelihood_penalty'], abs=2e-10)


def test_independent_joint_gradient_matches_finite_differences():
    objective = _JointObjective(load_example())
    point = np.array([.3, log(.7), .3, log(1.2), .4])
    _, gradient = objective(point)
    h = 1e-5
    finite_difference = []
    for j in range(len(point)):
        delta = np.zeros_like(point)
        delta[j] = h
        finite_difference.append((objective(point+delta)[0]-objective(point-delta)[0])/(2*h))
    assert_allclose(gradient, finite_difference, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("times", [
    (1e-300, 1e-180, 1e-60),
    (1-1e-8, 1-5e-9, 1-1e-9),
    (np.nextafter(0., 1.), .3, np.nextafter(1., 0.)),
])
@pytest.mark.parametrize("causes", [(1, 1, 2), (1, 1, 1), (2, 2, 2)])
def test_extreme_valid_failure_times_against_independent_joint_slsqp(times, causes):
    data = BPCSExperiment((make_block(causes, times),))
    analytical, numerical = fit_order_restricted_mle(data), fit_slsqp(data)
    assert analytical.converged and numerical.converged
    assert numerical.status == analytical.status
    assert abs(log(numerical.a/analytical.a)) < 3e-5
    B = sum(analytical.b_by_block['A'])
    assert_allclose(np.array(numerical.b_by_block['A'])/B,
                    np.array(analytical.b_by_block['A'])/B, rtol=3e-5, atol=1e-7)
    assert numerical.b_by_block['A'][0] <= numerical.b_by_block['A'][1]
    assert abs(numerical.loglikelihood-analytical.loglikelihood) < 1e-8
    assert numerical.diagnostics['projected_score_norm'] <= numerical.diagnostics['score_threshold']
    assert any(s['verified'] for s in numerical.diagnostics['starts'])


def test_nearly_coincident_times_report_range_failure_instead_of_finite_mle():
    data = BPCSExperiment((make_block((1, 1, 2), (.5, .50000001, .50000002)),))
    r = fit_order_restricted_mle(data)
    assert not r.converged and r.status == 'numerical_failure'
    assert r.a is None and not r.b_by_block


def test_independent_optimizer_budget_exhaustion_is_not_verified():
    result = fit_slsqp(load_example(), maxiter=1)
    assert not result.converged and result.status == 'optimization_failed'
    assert not any(run['verified'] for run in result.diagnostics['starts'])


def test_independent_optimizer_rejects_unbounded_design():
    data = BPCSExperiment((make_block((1,), (.3,)),))
    result = fit_slsqp(data)
    assert result.status == 'unbounded_likelihood' and not result.converged
    assert result.a is None and result.loglikelihood == np.inf


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_independent_optimizer_iteration_controls(bad):
    with pytest.raises(ValueError):
        fit_slsqp(load_example(), maxiter=bad)


@pytest.mark.parametrize("bad", [0, -1, True, np.inf, np.nan, '1'])
def test_independent_optimizer_score_controls(bad):
    with pytest.raises(ValueError):
        fit_slsqp(load_example(), score_tol=bad)


def test_reproducible_report_contains_independent_convergence_and_input_data():
    report = validation_report(load_example())
    assert len(report['input']) == 7
    assert report['independent_slsqp']['converged']
    assert report['comparison']['absolute_loglikelihood_difference'] < 1e-8
    assert report['comparison']['absolute_log_a_difference'] < 3e-5
    assert report['restricted']['diagnostics']['pooled_blocks'] == {'A': True, 'B': False}
