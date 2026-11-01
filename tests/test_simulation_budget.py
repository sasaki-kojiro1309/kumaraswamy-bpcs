"""Focused budget checks; no MCMC calls."""
import time
import pytest
from experiments.simulation_budget import ExecutionBudget, BudgetExpired
from experiments.simulation_runner import fit_stage
from experiments.simulation_config import load_config,ROOT
from experiments.simulation_storage import commit_json


def test_zero_budget_refuses_scheduling():
    b=ExecutionBudget(seconds=0,reserve_seconds=0)
    with pytest.raises(BudgetExpired):
        with b.supervise(): pytest.fail('scheduled expired work')


def test_inflight_timeout_bypasses_estimator_exception_handlers():
    budget=ExecutionBudget(seconds=.03,reserve_seconds=0)
    start=time.monotonic()
    with pytest.raises(BudgetExpired):
        with budget.supervise():
            try: time.sleep(.3)
            except Exception: pytest.fail('budget was swallowed by estimator handler')
    assert time.monotonic()-start<.25


def test_deadline_in_the_past_expires():
    assert ExecutionBudget(deadline_unix=time.time()-1).remaining()==0
