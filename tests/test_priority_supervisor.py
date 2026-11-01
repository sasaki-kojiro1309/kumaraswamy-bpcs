"""Quick scheduler regressions; no sampling or saved-study writes."""
from experiments import priority_supervisor as scheduler


def test_three_successes_precede_partial_and_unstarted(monkeypatch):
    states={1:{m:True for m in scheduler.BAYES[:2]},
            3:{m:True for m in scheduler.BAYES[:3]}}
    monkeypatch.setattr(scheduler,'state',lambda root,fid,sid,rep:states.get(rep,{}))
    assert scheduler.select_dataset(None,None,'violation_late',5)==3


def test_never_retry_a_diagnosed_failure_or_complete_dataset(monkeypatch):
    states={1:{m:True for m in scheduler.BAYES},
            2:{scheduler.BAYES[0]:True,scheduler.BAYES[-1]:False}}
    monkeypatch.setattr(scheduler,'state',lambda root,fid,sid,rep:states.get(rep,{}))
    assert scheduler.select_dataset(None,None,'violation_late',3)==3


def test_no_work_when_every_dataset_is_complete_or_failed(monkeypatch):
    monkeypatch.setattr(scheduler,'state',lambda root,fid,sid,rep:{scheduler.BAYES[-1]:False})
    assert scheduler.select_dataset(None,None,'strict_order_late',20) is None
