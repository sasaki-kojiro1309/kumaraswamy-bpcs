"""Reallocate one worker after persistent violation convergence failures."""
from pathlib import Path
import time
from experiments.legacy.priority_supervisor import state,fit_exact,BAYES
from experiments.simulation_runner import campaign
from experiments.simulation_config import load_config
from experiments.simulation_budget import ExecutionBudget,BudgetExpired
from experiments.simulation_storage import read_json,commit_json


def main():
    cfg=load_config(Path('configs/simulation_submission.json'));root=Path(cfg['output_dir']).resolve();fid=campaign(root,cfg)
    logs=sorted(root.glob('diagnostics/priority_supervision/*/started.json'))[-1].parent
    cap,_=read_json(logs/'external_cap_started.json')
    budget=ExecutionBudget();budget.deadline=cap['cap_monotonic'];budget.reserve_seconds=8
    events=[]
    commit_json(logs/'assist_started.json',dict(started_unix=time.time(),deadline_monotonic=budget.deadline,
        reason='Repeated violation failures without new pairs; allocate released third worker to additional complete comparisons.',
        no_violation_failure_retried=True))
    try:
        while budget.remaining()>0:
            candidates=[]
            for scenario in cfg['scenarios']:
                if scenario['id']=='violation_late':continue
                states=[state(root,fid,scenario['id'],r) for r in range(1,21)]
                frontier=next((r for r,s in enumerate(states,1) if len(s)<4 and all(s.values())),None)
                if frontier is None:continue
                # Stay three whole datasets ahead of that scenario's original
                # worker, avoiding its active/next tasks. Every task is still
                # immutable and protected by the established task lock.
                rep=next((r for r in range(frontier+3,21) if len(states[r-1])<4 and all(states[r-1].values())),None)
                if rep is not None:
                    complete=sum(len(s)==4 and all(s.values()) for s in states)
                    candidates.append((complete,rep,scenario['id'],scenario))
            if not candidates:break
            _,rep,_,scenario=min(candidates,key=lambda x:x[:3])
            current=state(root,fid,scenario['id'],rep)
            missing=[m for m in BAYES if m not in current]
            missing.sort(key=lambda m:(m!='bayes_hierarchical_within_block',BAYES.index(m)))
            for method in missing:
                if budget.remaining()<=0:raise BudgetExpired()
                record=fit_exact(root,cfg,scenario,rep,method,fid,budget)
                event=dict(scenario_id=scenario['id'],repetition=rep,method=method,status=record['status'],
                    runtime_seconds=record['runtime_seconds'],remaining_seconds=budget.remaining())
                events.append(event);commit_json(logs/'assist_events'/f"{scenario['id']}_{rep:03d}_{method}.json",event)
                print(event,flush=True)
                if not record['fit_success']:break
    except BudgetExpired:pass
    finally:
        commit_json(logs/'assist_finished.json',dict(finished_unix=time.time(),events=events))


if __name__=='__main__':main()
