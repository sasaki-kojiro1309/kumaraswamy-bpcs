"""Independent algebra, sequential-risk, and sampling-density checks for Stage 3."""

from itertools import product
from math import log, prod

import numpy as np
import pytest
from scipy.integrate import quad

from experiments.stage2_example import load_example
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.likelihood import (block_likelihood, block_loglikelihood, likelihood,
                            log_censoring_constant, loglikelihood)
from src.model import CompetingRisks, Kumaraswamy


def block_with(n, r, times, causes, label="A"):
    return BlockData(BlockScheme(label, n, len(r), r), tuple(
        Observation(label, t, c, removal) for t, c, removal in zip(times, causes, r)
    ))


def test_manual_uniform_minimum_case():
    # a=1,B=1 => minimum uniform, g1=1/4,g2=3/4,S(t)=1-t.
    block = block_with(3, (1, 0), (.2, .6), (1, 2))
    # C=3*1; L=C*(1/4)*(3/4)*(1-.2)=.45 exactly by hand.
    assert block_likelihood(block, 1, .25, .75) == pytest.approx(.45)
    assert block_loglikelihood(block, 1, .25, .75) == pytest.approx(log(.45))
    assert block_likelihood(block, 1, .25, .75, include_constant=False) == pytest.approx(.15)


@pytest.mark.parametrize("a,b1,b2", [(2, 1, 2), (.7, .4, 1.1), (3, 2, .3)])
def test_block_against_closed_form_latent_and_conditional_risk_formulations(a, b1, b2):
    for block in load_example().blocks:
        times = np.array([row.time for row in block.observations])
        r = np.array(block.scheme.withdrawals)
        counts = block.cause_counts
        B = b1+b2
        logq = np.log1p(-times**a)  # independent, moderate-value algebra oracle
        logc = log(prod(block.scheme.risk_sets))
        expanded = (logc + block.scheme.m*log(a) + counts[1]*log(b1)
                    + counts[2]*log(b2) + (a-1)*sum(np.log(times))
                    + sum(((r+1)*B-1)*logq))
        latent = {1: Kumaraswamy(a, b1), 2: Kumaraswamy(a, b2)}
        S = lambda t: (1-t**a)**B
        raw = prod(block.scheme.risk_sets)
        sequential = 0.0
        previous = 0.0
        for row, risk in zip(block.observations, block.scheme.risk_sets):
            g = latent[row.cause].pdf(row.time) * latent[3-row.cause].sf(row.time)
            raw *= g*S(row.time)**row.withdrawals
            # Conditional density of NEXT failure among the current risk set.
            sequential += (log(risk)+log(g)+(risk-1)*log(S(row.time))
                           -risk*log(S(previous)))
            previous = row.time
        actual = block_loglikelihood(block, a, b1, b2)
        assert actual == pytest.approx(expanded, abs=2e-12)
        assert actual == pytest.approx(log(raw), abs=2e-12)
        assert actual == pytest.approx(sequential, abs=2e-12)


def test_joint_factorizes_and_parameters_match_block_ids_not_position():
    data = load_example()
    params = {"B": (.8, 1.2), "A": (1, 2)}  # intentionally reversed
    values = [block_loglikelihood(b, 2, *params[b.scheme.block]) for b in data.blocks]
    assert loglikelihood(data, 2, params) == pytest.approx(sum(values))
    assert likelihood(data, 2, params) == pytest.approx(prod(np.exp(values)))
    reverse_data = BPCSExperiment(tuple(reversed(data.blocks)))
    assert loglikelihood(reverse_data, 2, params) == loglikelihood(data, 2, params)


@pytest.mark.parametrize("a,pairs", [
    (2, {"A": (1, 2), "B": (.8, 1.2)}),
    (.5, {"A": (.2, .3), "B": (3, 4)}),
    (4, {"A": (2, .1), "B": (1, 1)}),
])
def test_censoring_constant_is_parameter_independent_and_not_a_cause_multinomial(a, pairs):
    data = load_example()
    # A:6*4*3=72; B:7*6*4*3=504; joint=36288.
    assert log_censoring_constant(data.blocks[0]) == pytest.approx(log(72))
    assert log_censoring_constant(data.blocks[1]) == pytest.approx(log(504))
    assert log_censoring_constant(data) == pytest.approx(log(36288))
    full = loglikelihood(data, a, pairs)
    kernel = loglikelihood(data, a, pairs, include_constant=False)
    assert full-kernel == pytest.approx(log(36288))
    assert likelihood(data, a, pairs)/likelihood(data, a, pairs, include_constant=False) == pytest.approx(36288)


def test_sum_over_all_cause_sequences_recovers_time_only_density():
    for template in load_example().blocks:
        a, b1, b2 = 1.3, .7, 1.1
        times = tuple(row.time for row in template.observations)
        total = 0.0
        for causes in product((1, 2), repeat=template.scheme.m):
            block = block_with(template.scheme.n, template.scheme.withdrawals, times, causes)
            total += block_likelihood(block, a, b1, b2)
        minimum = Kumaraswamy(a, b1+b2)
        expected = prod(template.scheme.risk_sets) * prod(
            minimum.pdf(t)*minimum.sf(t)**r
            for t, r in zip(times, template.scheme.withdrawals)
        )
        assert total == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("n,r,a,b1,b2", [
    (3, (1, 0), 1, .25, .75),  # genuinely progressive
    (3, (0, 1), 1, .25, .75),  # Type-II
    (2, (0, 0), 1, .25, .75),  # complete
    (3, (1, 0), 2, .7, 1.3),   # nonuniform minimum
])
def test_full_density_integrates_over_ordered_times_and_causes(n, r, a, b1, b2):
    def marginalized(t1, t2):
        return sum(block_likelihood(block_with(n, r, (t1, t2), causes), a, b1, b2)
                   for causes in product((1, 2), repeat=2))
    mass = quad(lambda t1: quad(lambda t2: marginalized(t1, t2), t1, 1,
                                epsabs=1e-9)[0], 0, 1, epsabs=1e-9)[0]
    assert mass == pytest.approx(1, abs=1e-8)


def test_first_failure_density_integrals_and_cause_probability():
    # With m=1 all n-1 survivors are withdrawn at the first failure.
    n, a, b1, b2 = 4, 1.5, .7, 1.3
    masses = []
    for cause in (1, 2):
        mass = quad(lambda t: block_likelihood(
            block_with(n, (n-1,), (t,), (cause,)), a, b1, b2), 0, 1)[0]
        masses.append(mass)
    assert masses == pytest.approx([b1/(b1+b2), b2/(b1+b2)], abs=1e-9)
    assert sum(masses) == pytest.approx(1)


def test_single_cause_counts_and_cause_relabeling():
    original = block_with(3, (1, 0), (.2, .6), (1, 1))
    relabeled = block_with(3, (1, 0), (.2, .6), (2, 2))
    assert np.isfinite(block_loglikelihood(original, 2, .7, 1.4))
    assert block_loglikelihood(original, 2, .7, 1.4) == pytest.approx(
        block_loglikelihood(relabeled, 2, 1.4, .7))


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad", [0, -1, np.inf, -np.inf, np.nan, True, "2", [1], 1+1j])
def test_invalid_parameters_rejected_in_block_and_experiment(position, bad):
    values = [2, 1, 2]
    values[position] = bad
    block = load_example().blocks[0]
    with pytest.raises(ValueError):
        block_loglikelihood(block, *values)
    with pytest.raises(ValueError):
        loglikelihood(BPCSExperiment((block,)), values[0], {"A": values[1:]})


@pytest.mark.parametrize("params", [None, [], {}, {"A": (1, 2)},
    {"A": (1, 2), "B": (1, 2), "C": (1, 2)},
    {"A": (1,), "B": (1, 2)}, {"A": (1, 2, 3), "B": (1, 2)},
    {"A": None, "B": (1, 2)}, {"A": "12", "B": (1, 2)},
    {"A": {1: 2, 2: 3}, "B": (1, 2)}])
def test_invalid_parameter_mappings(params):
    with pytest.raises(ValueError):
        loglikelihood(load_example(), 2, params)


def test_invalid_containers_flags_and_overflowing_combined_parameter():
    data = load_example()
    with pytest.raises(ValueError, match="BlockData"):
        block_loglikelihood(data, 2, 1, 2)
    with pytest.raises(ValueError, match="BPCSExperiment"):
        loglikelihood(data.blocks[0], 2, {"A": (1, 2)})
    with pytest.raises(ValueError):
        log_censoring_constant(None)
    with pytest.raises(ValueError, match="boolean"):
        block_loglikelihood(data.blocks[0], 2, 1, 2, include_constant="False")
    with pytest.raises(ValueError, match="finite"):
        block_loglikelihood(data.blocks[0], 2, 1e308, 1e308)


def test_logarithms_retain_likelihood_when_ordinary_likelihood_underflows():
    block = block_with(2, (1,), (.99,), (1,))
    logvalue = block_loglikelihood(block, 2, 400, 600)
    # Independently expanded single-row expression: C=2, r+1=2, B=1000.
    expected = log(2)+log(2)+log(400)+log(.99)+(2000-1)*np.log1p(-.99**2)
    assert logvalue == pytest.approx(expected)
    assert np.isfinite(logvalue)
    assert block_likelihood(block, 2, 400, 600) == 0


def test_ordinary_density_overflow_is_not_clipped_and_logs_remain_usable():
    block = block_with(2, (0, 0), (1e-310, 2e-310), (1, 1))
    assert np.isfinite(block_loglikelihood(block, 1, 1e308, 1))
    assert block_likelihood(block, 1, 1e308, 1) == np.inf


def test_unrepresentable_log_arithmetic_is_reported_separately():
    block = block_with(2, (1,), (.999999,), (1,))
    with pytest.raises(FloatingPointError):
        block_loglikelihood(block, 1, 5e307, 5e307)
