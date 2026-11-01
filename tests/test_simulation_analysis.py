"""Metric/schema tests written but not executed during implementation."""
import pytest

from experiments.simulation_adapters import _interval_record
from experiments.simulation_analysis import point_metrics, interval_metrics, paired_comparison
from experiments.simulation_storage import CompatibilityError
from src.uncertainty import Interval


def point(uid,value,truth=2.,valid=True):
    return dict(dataset_id=uid,dataset_sha256='hash_'+uid,estimate=value,truth=truth,valid_point=valid)


def test_AB_is_absolute_signed_bias_not_mean_absolute_error():
    rows=[point('a',1.),point('b',3.),point('c',None,valid=False)]
    metrics=point_metrics(rows)
    assert metrics['R_generated']==3 and metrics['R_point']==2
    assert metrics['absolute_bias']==0 and metrics['mse']==1
    assert metrics['signed_bias']==0 and metrics['mean_estimate']==2
    assert metrics['point_availability_rate']==pytest.approx(2/3)
    assert metrics['estimate_sd']==pytest.approx(2**.5)


def test_empty_exact_set_has_zero_coverage_and_length():
    empty=_interval_record('p[A]',.25,Interval(None,None,'exact',.95,'empty'),'exact','confidence',.95)
    wide=_interval_record('p[A]',.25,Interval(0.,.5,'exact',.95),'exact','confidence',.95)
    missing=_interval_record('p[A]',.25,None,'exact','confidence',.95)
    out=interval_metrics([empty,wide,missing])
    assert out['R_generated']==3 and out['R_CP']==out['R_AW']==2
    assert out['coverage_probability']==.5 and out['average_width']==.25
    assert out['empty_set_count']==1 and out['unavailable_count']==1
    assert out['interval_availability_rate']==pytest.approx(2/3)


def test_exploratory_credible_interval_excluded_but_preserved():
    raw={'lower':.1,'upper':.8,'probability':.95,'empirical_mass':.95,'method':'actual posterior interval'}
    row=_interval_record('b1[A]',.5,raw,'posterior_shortest','credible',.95,usable=False)
    assert row['available'] and row['covered']==1 and not row['valid_interval']
    assert row['interval_kind']=='credible' and row['empirical_posterior_mass']==.95
    assert interval_metrics([row])['R_CP']==0


def test_invalid_endpoints_do_not_enter_interval_denominator():
    row=_interval_record('a',1.,{'lower':3.,'upper':1.},'log','confidence',.95)
    assert not row['valid_interval'] and interval_metrics([row])['R_AW']==0


def test_paired_mse_uses_common_success_ids_only():
    U=[point('1',3.),point('2',5.),point('3',2.)]
    R=[point('1',2.),point('2',3.),point('3',None,valid=False)]
    out=paired_comparison(U,R)
    assert out['R_U']==3 and out['R_R']==out['R_paired']==2
    assert out['MSE_U_paired']==5 and out['MSE_R_paired']==.5
    assert out['relative_mse_reduction_percent']==90
    assert out['mean_squared_error_difference_U_minus_R']==4.5
    assert out['paired_difference_mcse']==pytest.approx(3.5)
    assert not out['success_sets_equal']


def test_pairing_checks_truth_and_data_hash():
    U=[point('1',3.)];R=[point('1',2.)];R[0]['dataset_sha256']='different'
    with pytest.raises(CompatibilityError):paired_comparison(U,R)
    with pytest.raises(CompatibilityError):paired_comparison(U,[point('1',2.,truth=3.)])


def test_zero_ratio_and_empty_summary_are_explicit():
    out=paired_comparison([point('1',2.)],[point('1',2.)])
    assert out['relative_mse_reduction_percent'] is None and out['R_paired']==1
    assert out['paired_difference_mcse'] is None
    assert point_metrics([])['R_point']==0
    assert interval_metrics([])['coverage_probability'] is None
