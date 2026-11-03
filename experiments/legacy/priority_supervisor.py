"""Deadline-bounded paired-dataset scheduling, using existing fit/storage APIs."""
import multiprocessing as mp
from pathlib import Path
import time

from experiments.simulation_config import load_config, METHODS, seed_for, digest
from experiments.simulation_storage import read_json, commit_json, task_lock, save_draws
from experiments.simulation_generator import dataset_path, load_dataset
from experiments.simulation_runner import campaign, result_path, verify_result
from experiments.simulation_adapters import fit_method
from experiments.simulation_budget import ExecutionBudget, BudgetExpired

BAYES=tuple(METHODS[2:])
# The user's 30 minutes start with the first execution timestamp in this turn.
START_UNIX=1793551660.0


def state(root,fit_id,sid,rep):
    result={}
    for method in BAYES:
        path=result_path(root,fit_id,sid,rep,method)
        if path.exists():
            record,_=read_json(path)
            result[method]=record['fit_success']
    return result


def select_dataset(root,fit_id,sid,repetitions):
    """Never revisit failed datasets; rank by existing successful methods."""
    candidates=[]
    for rep in range(1,repetitions+1):
        current=state(root,fit_id,sid,rep)
        if len(current)==4 or any(not success for success in current.values()):continue
        candidates.append((-sum(current.values()),rep))
    return min(candidates)[1] if candidates else None


def fit_exact(root,cfg,scenario,rep,method,fit_id,budget):
    """Same immutable commit/identity/seed contract as simulation_runner.fit_stage.

    Only scheduling differs: this function fits one exact saved dataset, avoiding
    unrelated earlier repetitions that cannot become complete paired datasets.
    """
    path=dataset_path(root,scenario['id'],rep)
    data,metadata,sha=load_dataset(path,scenario=scenario,repetition=rep,master_seed=cfg['master_seed'])
    output=result_path(root,fit_id,scenario['id'],rep,method)
    if output.exists():return verify_result(output,fit_id,metadata,sha,method)[0]
    lock=Path(root)/'metadata'/'locks'/'fitting'/fit_id/f"{metadata['dataset_id']}_{method}.lock"
    with task_lock(lock):
        if output.exists():return verify_result(output,fit_id,metadata,sha,method)[0]
        seed=seed_for(cfg['mcmc_master_seed'],'kw-bpcs-mcmc-v1',metadata['dataset_id'],method)
        try:
            with budget.supervise():bundle,draws=fit_method(data,scenario,cfg,method,seed)
        except BudgetExpired:
            commit_json(Path(root)/'diagnostics'/'budget_interruptions'/fit_id/
                f"{metadata['dataset_id']}_{method}_{time.time_ns()}.json",
                dict(dataset_id=metadata['dataset_id'],dataset_sha256=sha,scenario_id=scenario['id'],
                    repetition=rep,method=method,status='budget_interrupted',fit_id=fit_id,mcmc_seed=seed))
            raise
        bundle.update(fit_id=fit_id,dataset_id=metadata['dataset_id'],dataset_sha256=sha,
            dataset_path=str(path.relative_to(root)),scenario_id=scenario['id'],repetition=rep,
            ordering=scenario['ordering'],censoring=scenario['censoring'],
            scenario_definition_hash=digest(scenario),draw_artifact=None)
        save=cfg['save_draws']
        selected=(rep in save['repetitions'] or save['diagnostic_failures'] and bundle['status']=='diagnostics_failed'
                  or save['numerical_failures'] and bundle['status']=='numerical_failure')
        if selected and draws is not None:
            names,array=draws
            target,drawsha=save_draws(Path(root)/'draws'/fit_id/metadata['dataset_id']/method,names,array)
            bundle['draw_artifact']=dict(path=str(target.relative_to(root)),sha256=drawsha,names=list(names),shape=list(array.shape))
        commit_json(output,bundle)
        return bundle


def worker(cfg,scenario,root,fit_id,deadline,logs):
    budget=ExecutionBudget();budget.deadline=deadline;budget.reserve_seconds=8
    completed=[]
    try:
        while budget.remaining()>0:
            rep=select_dataset(root,fit_id,scenario['id'],cfg['repetitions'])
            if rep is None:break
            current=state(root,fit_id,scenario['id'],rep)
            missing=[m for m in BAYES if m not in current]
            # Close three-of-four first. Otherwise test the difficult constrained
            # hierarchy before spending time on a dataset it might invalidate.
            missing.sort(key=lambda m: (m!='bayes_hierarchical_within_block',BAYES.index(m)))
            for method in missing:
                if budget.remaining()<=0:raise BudgetExpired()
                record=fit_exact(Path(root),cfg,scenario,rep,method,fit_id,budget)
                event=dict(scenario_id=scenario['id'],repetition=rep,method=method,
                    status=record['status'],runtime_seconds=record['runtime_seconds'],remaining_seconds=budget.remaining())
                completed.append(event)
                commit_json(Path(logs)/'events'/f"{scenario['id']}_{rep:03d}_{method}.json",event)
                print(event,flush=True)
                if not record['fit_success']:break
    except BudgetExpired:pass
    finally:
        commit_json(Path(logs)/f"{scenario['id']}_finished.json",dict(events=completed,finished_monotonic=time.monotonic()))


def main():
    cfg=load_config(Path('configs/simulation_submission.json'));root=Path(cfg['output_dir']).resolve()
    fit_id=campaign(root,cfg);assert fit_id=='c81a97c91afc340fd144ab3f'
    # A single monotonic deadline is calibrated once, then passed unchanged to
    # every spawned child; no child creates a fresh 23/30-minute allowance.
    now=time.monotonic();hard=now+(START_UNIX+1800-time.time());cutoff=hard-300
    assert cutoff>now
    logs=root/'diagnostics'/'priority_supervision'/str(time.time_ns())
    commit_json(logs/'started.json',dict(fit_id=fit_id,task_start_unix=START_UNIX,
        supervisor_started_unix=time.time(),hard_deadline_monotonic=hard,
        compute_deadline_monotonic=cutoff,hard_deadline_unix=START_UNIX+1800,
        compute_deadline_unix=START_UNIX+1500,workers=3,
        policy='three-of-four first; within each scenario prioritize existing successes, then constrained hierarchy first; never retry diagnosed failures'))
    ctx=mp.get_context('spawn');children=[]
    # Violation is dispatched first; each scenario has one dedicated worker so
    # completing difficult violation fits does not starve the other scenarios.
    for scenario in sorted(cfg['scenarios'],key=lambda s:s['id']!='violation_late'):
        child=ctx.Process(target=worker,args=(cfg,scenario,str(root),fit_id,cutoff,str(logs)))
        child.start();children.append(child)
    while any(c.is_alive() for c in children) and time.monotonic()<cutoff:time.sleep(.5)
    terminated=[]
    for i,child in enumerate(children):
        if child.is_alive():child.terminate();terminated.append(i)
    for child in children:
        child.join(timeout=2)
        if child.is_alive():child.kill();child.join(timeout=2)
    commit_json(logs/'finished.json',dict(finished_unix=time.time(),finished_monotonic=time.monotonic(),
        exit_codes=[c.exitcode for c in children],externally_terminated_workers=terminated,
        hard_deadline_monotonic=hard,compute_deadline_monotonic=cutoff))
    print('Priority supervisor finished',logs,flush=True)


if __name__=='__main__':main()
