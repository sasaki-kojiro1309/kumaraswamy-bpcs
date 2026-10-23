"""Accounting, malformed designs, ordering, and block isolation for Stage 2."""

from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from experiments.stage2_example import load_example
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation


def small_block(label="A", times=(.2, .5), causes=(1, 2), removals=(0, 1)):
    scheme = BlockScheme(label, 3, 2, (0, 1))
    return BlockData(scheme, tuple(
        Observation(label, t, c, r) for t, c, r in zip(times, causes, removals)
    ))


def test_two_block_example_matches_hand_calculated_accounting():
    experiment = load_example()
    frame = experiment.to_frame()
    assert list(frame.block) == ["A"]*3 + ["B"]*4
    assert list(frame.failure_index) == [1, 2, 3, 1, 2, 3, 4]
    assert list(frame.at_risk) == [6, 4, 3, 7, 6, 4, 3]
    assert list(frame.after_failure) == [5, 3, 2, 6, 5, 3, 2]
    assert list(frame.remaining) == [4, 3, 0, 6, 4, 3, 0]
    assert list(frame.withdrawals) == [1, 0, 2, 0, 1, 0, 2]
    assert list(frame.time) == [.15, .4, .7, .1, .25, .55, .8]
    summary = experiment.summary_frame()
    assert summary.to_dict("records") == [
        {"block": "A", "n": 6, "m": 3, "withdrawn": 3, "cause_1": 2, "cause_2": 1},
        {"block": "B", "n": 7, "m": 4, "withdrawn": 3, "cause_1": 1, "cause_2": 3},
    ]
    assert summary.n.sum() == 13
    assert summary.m.sum() == 7
    assert summary.withdrawn.sum() == 6


@pytest.mark.parametrize("n,m,r,risks", [
    (3, 3, (0, 0, 0), (3, 2, 1)),  # complete observation
    (5, 3, (0, 0, 2), (5, 4, 3)),  # ordinary Type-II censoring
    (5, 1, (4,), (5,)),             # one failure, all survivors withdrawn
    (1, 1, (0,), (1,)),
    (5, 3, (2, 0, 0), (5, 2, 1)),  # early withdrawals
])
def test_special_case_schedules(n, m, r, risks):
    assert BlockScheme("A", n, m, r).risk_sets == risks


def test_all_small_complete_schedules_preserve_risk_sets():
    # Enumerate all nonnegative schedules with the right total for n<=6.
    # Compare the implementation to the closed-form risk-set identity.
    for n in range(1, 7):
        for m in range(1, n+1):
            for removals in product(range(n-m+1), repeat=m):
                if sum(removals) != n-m:
                    continue
                scheme = BlockScheme("A", n, m, removals)
                expected = tuple(n-j-sum(removals[:j]) for j in range(m))
                assert scheme.risk_sets == expected
                assert all(risk >= 1+r for risk, r in zip(expected, removals))
                assert expected[-1] - 1 - removals[-1] == 0


@pytest.mark.parametrize("n,m,r,message", [
    (0, 1, (0,), "n must"),
    (3, 0, (), "m must"),
    (2, 3, (0, 0, 0), "m cannot exceed"),
    (3.0, 2, (0, 1), "n must"),
    (3, 2.0, (0, 1), "m must"),
    (True, 1, (0,), "n must"),
    (3, 2, (1,), "exactly m"),
    (3, 2, (0, 0, 1), "exactly m"),
    (3, 2, (0, -1), "withdrawal count"),
    (3, 2, (0, 1.0), "withdrawal count"),
    (3, 2, (0, True), "withdrawal count"),
    (3, 2, None, "iterable"),
    (3, 2, (0, 0), "n = m"),
    (3, 2, (0, 3), "exceed survivors"),
    (3, 2, (2, 0), "no unit remains"),
    (3, 2, (3, 0), "exceed survivors"),
])
def test_invalid_schedules(n, m, r, message):
    with pytest.raises(ValueError, match=message):
        BlockScheme("A", n, m, r)


@pytest.mark.parametrize("label", ["", "   ", None, 1])
def test_invalid_block_ids(label):
    with pytest.raises(ValueError, match="block"):
        BlockScheme(label, 1, 1, (0,))
    with pytest.raises(ValueError, match="block"):
        Observation(label, .2, 1, 0)


@pytest.mark.parametrize("time", [0, 1, -.1, 1.1, np.nan, np.inf, -np.inf, True, ".2"])
def test_invalid_failure_times(time):
    with pytest.raises(ValueError, match="failure time"):
        Observation("A", time, 1, 0)


@pytest.mark.parametrize("cause", [0, 3, -1, True, 1.0, "1", np.nan])
def test_invalid_cause_codes(cause):
    with pytest.raises(ValueError, match="cause"):
        Observation("A", .2, cause, 0)


@pytest.mark.parametrize("withdrawals", [-1, .5, True, "1", np.nan])
def test_invalid_observation_withdrawals(withdrawals):
    with pytest.raises(ValueError, match="withdrawals"):
        Observation("A", .2, 1, withdrawals)


def test_missing_cause_representable_but_rejected_by_baseline():
    row = Observation("A", .2, None, 0)
    assert row.cause is None
    with pytest.raises(ValueError, match="unknown causes"):
        BlockData(BlockScheme("A", 1, 1, (0,)), (row,))


@pytest.mark.parametrize("times", [(.2, .2), (.5, .2)])
def test_tied_or_decreasing_times_rejected_without_sorting(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        small_block(times=times)


def test_observations_must_match_block_scheme_and_failure_target():
    scheme = BlockScheme("A", 3, 2, (0, 1))
    with pytest.raises(ValueError, match="exactly m"):
        BlockData(scheme, (Observation("A", .2, 1, 0),))
    with pytest.raises(ValueError, match="different block ID"):
        BlockData(scheme, (Observation("B", .2, 1, 0), Observation("A", .5, 2, 1)))
    # Equal total withdrawals is insufficient: removal must match each failure.
    with pytest.raises(ValueError, match="do not match"):
        small_block(removals=(1, 0))


def test_any_block_count_cross_block_overlap_and_single_cause_blocks():
    # A has no observed cause-2 failures, still a valid data set.
    first = small_block(causes=(1, 1))
    second = small_block("B", times=(.1, .3))
    third = small_block("C", times=(.2, .5))
    assert len(BPCSExperiment((first,)).blocks) == 1
    experiment = BPCSExperiment((first, second, third))
    assert len(experiment.blocks) == 3
    assert first.cause_counts == {1: 2, 2: 0}
    assert list(experiment.to_frame().time) == [.2, .5, .1, .3, .2, .5]
    with pytest.raises(ValueError, match="unique"):
        BPCSExperiment((first, first))
    with pytest.raises(ValueError, match="at least one"):
        BPCSExperiment(())


def test_immutable_data_detaches_input_lists_and_exported_tables():
    removals = [0, 1]
    scheme = BlockScheme("A", np.int64(3), np.int64(2), removals)
    removals[0] = 99
    assert scheme.withdrawals == (0, 1)
    rows = [Observation("A", .2, np.int64(1), 0), Observation("A", .5, 2, 1)]
    block = BlockData(scheme, rows)
    rows.clear()
    blocks = [block]
    experiment = BPCSExperiment(blocks)
    blocks.clear()
    frame = experiment.to_frame()
    frame.loc[0, "time"] = .9
    assert experiment.blocks[0].observations[0].time == .2
    with pytest.raises(FrozenInstanceError):
        scheme.n = 20


def test_wrong_container_types_raise_clear_errors():
    with pytest.raises(ValueError, match="BlockScheme"):
        BlockData(None, ())
    scheme = BlockScheme("A", 1, 1, (0,))
    with pytest.raises(ValueError, match="iterable"):
        BlockData(scheme, None)
    with pytest.raises(ValueError, match="Observation"):
        BlockData(scheme, ({"time": .2},))
    with pytest.raises(ValueError, match="iterable"):
        BPCSExperiment(None)
    with pytest.raises(ValueError, match="BlockData"):
        BPCSExperiment((scheme,))
