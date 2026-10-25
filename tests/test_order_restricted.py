"""Analytical restricted fit versus independent joint SLSQP optimization."""

from dataclasses import replace
from math import log

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.special import xlogy

from experiments.stage2_example import load_example
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.likelihood import loglikelihood
from src.mle import MLEResult, fit_mle, profile_loglikelihood
from src.order_restricted import fit, fit_order_restricted_mle


def block(label, causes, times=None):
    if times is None:
        times = np.linspace(.12, .83, len(causes))
    removals = tuple(j % 2 for j in range(len(causes)))
    scheme = BlockScheme(label, len(causes)+sum(removals), len(causes), removals)
    return BlockData(scheme, tuple(Observation(label, t, c, r)
                                  for t, c, r in zip(times, causes, removals)))


def numerical_fit(data):
    """Independent joint fit in (log a, b11,b12,...), allowing exact zeros.

    Uses the expanded row likelihood, finite-difference gradients, raw b bounds
    and linear order constraints. No profile, count/W update, fitted start,
    active-set choice, or restricted-estimator call is used in this oracle.
    """
    def objective(v):
        a = np.exp(v[0])
        total = 0.0
        for i, b in enumerate(data.blocks):
            b1, b2 = v[1+2*i:3+2*i]
            d = b.cause_counts
            total += xlogy(d[1], b1)+xlogy(d[2], b2)
            for row, risk in zip(b.observations, b.scheme.risk_sets):
                logq = np.log(-np.expm1(a*log(row.time)))
                total += (log(risk)+log(a)+(a-1)*log(row.time)
                          +((row.withdrawals+1)*(b1+b2)-1)*logq)
        return -float(total)

    k = len(data.blocks)
    constraints = {"type": "ineq",
                   "fun": lambda v: v[2::2]-v[1::2]}
    runs = [minimize(objective, np.array([log(a), *([b, 1.4*b]*k)]),
                     method="SLSQP", constraints=constraints,
                     bounds=[(-15, 15)]+[(0, None)]*(2*k),
                     options={"maxiter": 1000, "ftol": 1e-12})
            for a, b in [(.6, .4), (2., 1.)]]
    successful = [r for r in runs if r.success and np.isfinite(r.fun)]
    assert successful, [r.message for r in runs]
    r = min(successful, key=lambda r: r.fun)
    assert np.min(constraints['fun'](r.x)) >= -1e-9
    return np.exp(r.x[0]), r.x[1:].reshape(k, 2), -r.fun


@pytest.mark.parametrize("causes,active,pooled", [
    ((1, 2, 2), False, False),
    ((1, 1, 2), True, True),
    ((1, 2, 1, 2), True, False),
    ((1, 1, 1), True, True),
    ((2, 2, 2), False, False),
])
def test_single_block_cases_against_independent_joint_optimization(causes, active, pooled):
    data = BPCSExperiment((block('one', causes),))
    u, r = fit_mle(data), fit_order_restricted_mle(data)
    assert r.converged
    assert isinstance(r, MLEResult)
    assert r.diagnostics['active_constraints'] == {'one': active}
    assert r.diagnostics['pooled_blocks'] == {'one': pooled}
    assert r.a == u.a
    assert sum(r.b_by_block['one']) == pytest.approx(sum(u.b_by_block['one']))
    assert r.loglikelihood <= u.loglikelihood+1e-12
    na, nb, nl = numerical_fit(data)
    assert na == pytest.approx(r.a, rel=3e-5)
    assert_allclose(nb[0], r.b_by_block['one'], rtol=3e-5, atol=1e-7)
    assert nl == pytest.approx(r.loglikelihood, abs=1e-9)
    if active:
        assert r.b_by_block['one'][0] == r.b_by_block['one'][1] > 0
    elif 1 in causes:
        assert r.b_by_block == u.b_by_block
        assert r.loglikelihood == u.loglikelihood


def test_teaching_example_values_and_same_likelihood():
    data = load_example()
    u, r = fit_mle(data), fit_order_restricted_mle(data)
    assert r.converged and r.status == 'interior_mle'
    assert r.a == u.a == pytest.approx(1.173959840838)
    assert_allclose(r.b_by_block['A'], (.388281082470, .388281082470), atol=1e-12)
    assert r.b_by_block['B'] == u.b_by_block['B']
    assert r.diagnostics['active_constraints'] == {'A': True, 'B': False}
    assert r.loglikelihood == pytest.approx(.900779910541690, abs=1e-12)
    assert r.loglikelihood == loglikelihood(data, r.a, r.b_by_block)
    # Only A changes: 2 log(2/3)+log(1/3)-3 log(1/2).
    penalty = 2*log(2/3)+log(1/3)-3*log(.5)
    assert u.loglikelihood-r.loglikelihood == pytest.approx(penalty, abs=1e-12)
    assert r.diagnostics['loglikelihood_penalty'] == pytest.approx(penalty)
    na, nb, nl = numerical_fit(data)
    assert na == pytest.approx(r.a, rel=3e-5)
    assert_allclose(nb, list(r.b_by_block.values()), rtol=3e-5)
    assert nl == pytest.approx(r.loglikelihood, abs=1e-9)


@pytest.mark.parametrize('k', [1, 3, 5])
def test_arbitrary_blocks_and_order_with_mixed_constraints(k):
    cases = [(1, 1, 2), (1, 2, 2), (1, 2, 1, 2), (1, 1, 1), (2, 2, 2)]
    data = BPCSExperiment(tuple(reversed([block(f'facility {j}', cases[j]) for j in range(k)])))
    u, r = fit_mle(data), fit_order_restricted_mle(data)
    assert r.converged and r.a == u.a
    for b in data.blocks:
        label = b.scheme.block
        pair = r.b_by_block[label]
        assert pair[0] <= pair[1]
        assert sum(pair) == pytest.approx(sum(u.b_by_block[label]), rel=1e-14)
        assert r.diagnostics['active_constraints'][label] == (b.cause_counts[1] >= b.cause_counts[2])
    na, nb, nl = numerical_fit(data)
    assert na == pytest.approx(r.a, rel=3e-5)
    assert_allclose(nb, list(r.b_by_block.values()), rtol=4e-5, atol=1e-7)
    assert nl == pytest.approx(r.loglikelihood, abs=1e-9)


def test_positive_equality_is_attained_even_with_absent_cause_2():
    data = BPCSExperiment((block('A', (1, 1, 1)),))
    u, r = fit_mle(data), fit_order_restricted_mle(data)
    assert u.status == 'boundary_supremum'
    assert r.status == 'interior_mle' and r.converged
    assert r.diagnostics['boundary_parameters'] == ()
    assert r.diagnostics['active_constraints']['A']
    assert r.diagnostics['p_kkt_multipliers']['A'] == {'upper_p': 6., 'lower_p': 0.}
    assert u.loglikelihood-r.loglikelihood == pytest.approx(3*log(2))
    assert r.loglikelihood == loglikelihood(data, r.a, r.b_by_block)


def test_zero_cause_1_is_boundary_supremum_with_exact_linear_limit():
    data = BPCSExperiment((block('A', (2, 2, 2)),))
    r = fit_order_restricted_mle(data)
    assert r.status == 'boundary_supremum' and r.converged
    assert r.b_by_block['A'][0] == 0
    assert r.diagnostics['boundary_parameters'] == (('A', 1),)
    assert not r.diagnostics['active_constraints']['A']
    assert r.diagnostics['p_kkt_multipliers']['A'] == {'upper_p': 0., 'lower_p': 3.}
    W = 3/r.b_by_block['A'][1]
    for eps in (1e-2, 1e-5, 1e-8):
        interior = {'A': (eps, r.b_by_block['A'][1])}
        assert r.loglikelihood-loglikelihood(data, r.a, interior) == pytest.approx(eps*W, abs=1e-12)


def test_equality_counts_have_zero_order_multiplier():
    r = fit_order_restricted_mle(BPCSExperiment((block('A', (1, 2)),)))
    assert r.converged and r.diagnostics['active_constraints']['A']
    assert not r.diagnostics['pooled_blocks']['A']
    assert r.diagnostics['p_kkt_multipliers']['A']['upper_p'] == 0


def test_single_failure_mixed_with_identifying_block():
    data = BPCSExperiment((block('A', (1,), (.3,)), block('B', (1, 2, 2))))
    r = fit_order_restricted_mle(data)
    assert r.converged and r.status == 'interior_mle'
    assert r.b_by_block['A'][0] == r.b_by_block['A'][1]


@pytest.mark.parametrize('causes', [(1,), (2,), (1, 2)])
def test_unbounded_single_failure_blocks_remain_unbounded(causes):
    data = BPCSExperiment(tuple(block(str(j), (c,), (.2+.2*j,)) for j, c in enumerate(causes)))
    r = fit_order_restricted_mle(data)
    assert r.status == 'unbounded_likelihood' and not r.converged
    assert r.a is None and not r.b_by_block and r.loglikelihood == np.inf
    # Constant penalties cannot bound an increasing unbounded profile.
    penalty = sum(c == 1 for c in causes)*log(2)
    assert profile_loglikelihood(data, 100)-penalty > profile_loglikelihood(data, 10)-penalty


@pytest.mark.parametrize('options', [{'maxiter': 1}, {'initial_a': 1e-6, 'max_bracket_steps': 1}])
def test_solver_budget_failure_is_propagated(options):
    r = fit_order_restricted_mle(load_example(), **options)
    assert not r.converged and r.status == 'optimization_failed'
    assert r.a is None and not r.b_by_block


def test_numeric_range_failure_is_propagated():
    data = BPCSExperiment((block('A', (1, 2), (1e-300, 2e-300)),))
    r = fit_order_restricted_mle(data)
    assert not r.converged and r.status == 'numerical_failure'


@pytest.mark.parametrize('bad', [0, -1, np.nan, np.inf, True, '1', [1]])
@pytest.mark.parametrize('name', ['initial_a', 'score_tol', 'constraint_tol'])
def test_invalid_positive_controls(name, bad):
    with pytest.raises(ValueError):
        fit_order_restricted_mle(load_example(), **{name: bad})


@pytest.mark.parametrize('bad', [0, -1, True, 2.5])
@pytest.mark.parametrize('name', ['maxiter', 'max_bracket_steps'])
def test_invalid_iteration_controls(name, bad):
    with pytest.raises(ValueError):
        fit_order_restricted_mle(load_example(), **{name: bad})


def test_invalid_data_and_unsupported_restrictions():
    with pytest.raises(ValueError):
        fit_order_restricted_mle(None)
    for restriction in ('reverse', None, ['b_i1 <= b_i2']):
        with pytest.raises(ValueError):
            fit(load_example(), restriction=restriction)
    assert fit(load_example()).b_by_block == fit_order_restricted_mle(load_example()).b_by_block


def test_post_fit_arithmetic_failure_is_not_a_converged_fit(monkeypatch):
    import src.order_restricted as module
    def fail(*args, **kwargs):
        raise FloatingPointError('injected unrepresentable evaluation')
    monkeypatch.setattr(module, 'loglikelihood', fail)
    r = module.fit_order_restricted_mle(load_example())
    assert r.status == 'numerical_failure' and not r.converged
    assert r.a is None and not r.b_by_block


def test_objective_disagreement_is_not_reported_as_an_mle(monkeypatch):
    import src.order_restricted as module
    original = module.loglikelihood
    monkeypatch.setattr(module, 'loglikelihood', lambda *a, **kw: original(*a, **kw)+1)
    r = module.fit_order_restricted_mle(load_example())
    assert r.status == 'optimization_failed' and not r.converged


def test_restricted_fit_does_not_mutate_profile_result(monkeypatch):
    import src.order_restricted as module
    u = fit_mle(load_example())
    saved = replace(u, b_by_block=dict(u.b_by_block), diagnostics=dict(u.diagnostics))
    monkeypatch.setattr(module, 'fit_mle', lambda *a, **kw: u)
    module.fit_order_restricted_mle(load_example())
    assert u == saved
