"""Balanced POSIX parallel submission executor with a real shared hard deadline."""
import argparse
import datetime
import multiprocessing as mp
from pathlib import Path
import time

from experiments.simulation_budget import ExecutionBudget
from experiments.simulation_config import load_config,METHODS,digest
from experiments.simulation_runner import fit_stage,campaign
from experiments.simulation_storage import commit_json


def worker(index,cfg,scenario,root,deadline,logdir):
    methods=list(METHODS[2:]);offset=index%len(methods)
    methods=methods[offset:]+methods[:offset]
    budget=ExecutionBudget(deadline_unix=deadline,reserve_seconds=5.)
    completed=0;counts={}
    while budget.remaining()>0:
        fit_id,counts=fit_stage(Path(root),cfg,[scenario],methods,budget=budget,max_new_fits=1)
        completed+=counts['completed']
        print(f"worker={index} scenario={scenario['id']} new={completed} remaining={budget.remaining():.1f}s",flush=True)
        if counts['completed']==0 or counts['budget_exhausted']:
            break
    commit_json(Path(logdir)/f'worker_{index:02d}.json',dict(worker=index,scenario_id=scenario['id'],
                new_completed=completed,last_counts=counts,finished_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--workers-per-scenario',type=int,default=3)
    a=p.parse_args()
    if a.workers_per_scenario<1:raise ValueError('workers-per-scenario must be positive')
    cfg=load_config(a.config);root=Path(cfg['output_dir']).resolve()
    fit_id=campaign(root,cfg)
    ident=str(time.time_ns());logs=root/'diagnostics'/'supervision'/ident
    logs.mkdir(parents=True,exist_ok=False)
    commit_json(logs/'started.json',dict(fit_id=fit_id,deadline_unix=a.deadline_unix,
        started_unix=time.time(),workers_per_scenario=a.workers_per_scenario,
        config=cfg,planned_dataset_count=len(cfg['scenarios'])*cfg['repetitions']))
    context=mp.get_context('spawn');processes=[]
    for scenario in cfg['scenarios']:
        for j in range(a.workers_per_scenario):
            i=len(processes)
            process=context.Process(target=worker,args=(i,cfg,scenario,str(root),a.deadline_unix,str(logs)))
            process.start();processes.append(process)
    while any(process.is_alive() for process in processes) and time.time()<a.deadline_unix:
        time.sleep(.5)
    terminated=[]
    for i,process in enumerate(processes):
        if process.is_alive():
            process.terminate();terminated.append(i)
    for process in processes:process.join(timeout=3)
    commit_json(logs/'finished.json',dict(fit_id=fit_id,finished_unix=time.time(),
        exit_codes=[p.exitcode for p in processes],externally_terminated_workers=terminated,
        deadline_unix=a.deadline_unix))
    print('Supervisor finished',fit_id,'logs',logs,flush=True)


if __name__=='__main__':main()
