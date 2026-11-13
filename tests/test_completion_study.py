"""Regression checks for strict validity and identity-matched comparisons."""
from copy import deepcopy

import pytest

from experiments.completion_analysis import common_ids
from experiments.completion_study import acceptable, ROOT, OLD
from experiments.simulation_storage import read_json


def test_six_way_ids_are_intersected_not_matched_by_count():
    methods = ['one', 'two', 'three']
    rows = [dict(method=m, dataset_id=d, valid_point=True)
            for m, ids in [('one', ['A', 'B']), ('two', ['B', 'C']), ('three', ['B', 'D'])]
            for d in ids]
    assert common_ids(rows, methods) == ['B']
    rows[-2]['valid_point'] = False
    assert common_ids(rows, methods) == []


@pytest.fixture
def valid_saved():
    path = ROOT/'estimates'/'tasks'/OLD/'strict_order_late'/'rep_000001_bayes_hierarchical_within_block.json'
    return read_json(path)[0]


def test_saved_success_reused_only_after_strict_checks(valid_saved):
    assert acceptable(valid_saved)


@pytest.mark.parametrize('field,value', [('rhat', 1.011), ('ess_bulk', 399.9), ('ess_tail', 399.9), ('mcse_mean', None)])
def test_parameterwise_failure_cannot_be_marked_valid(valid_saved, field, value):
    record = deepcopy(valid_saved)
    record['bayesian_diagnostics'][-1][field] = value
    assert not acceptable(record)


def test_missing_primary_or_order_violation_rejected(valid_saved):
    record = deepcopy(valid_saved)
    record['points'] = [r for r in record['points'] if r['parameter'] != 'a']
    assert not acceptable(record)
    record = deepcopy(valid_saved)
    for row in record['points']:
        if row['parameter'] == 'b1[A]':
            row['estimate'] = 1e6
    assert not acceptable(record)
