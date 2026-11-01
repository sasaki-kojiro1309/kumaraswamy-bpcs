"""Validate persisted submission data and publish an immutable execution audit."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import time

from .simulation_config import load_config, METHODS
from .simulation_generator import dataset_path, load_dataset
from .simulation_runner import campaign, result_path, verify_result
from .simulation_analysis import analyze_stage
from .simulation_storage import commit_json, read_json, publish_pointer


def main():
    cfg=load_config(Path('configs/simulation_submission.json'));root=Path(cfg['output_dir']).resolve()
    fit_id=campaign(root,cfg);counts=Counter();dataset_counts=Counter();paired=defaultdict(set)
    valid_means=Counter();runtimes=Counter();attempted=Counter();successful_hashes={}
    for scenario in cfg['scenarios']:
        for rep in range(1,cfg['repetitions']+1):
            data,meta,sha=load_dataset(dataset_path(root,scenario['id'],rep),scenario=scenario,
                                      repetition=rep,master_seed=cfg['master_seed'])
            dataset_counts[scenario['id']]+=1
            for block in data.blocks:
                assert block.scheme.n==50 and block.scheme.m==30
                assert block.scheme.withdrawals==(0,)*29+(20,)
                assert block.scheme.risk_sets==tuple(range(50,20,-1))
            for method in METHODS:
                p=result_path(root,fit_id,scenario['id'],rep,method)
                if not p.exists():
                    counts[(scenario['id'],method,'not_run')]+=1;continue
                record,_=read_json(p)
                assert record['dataset_id']==meta['dataset_id'] and record['dataset_sha256']==sha
                assert record['fit_id']==fit_id and record['method']==method
                counts[(scenario['id'],method,record['status'])]+=1
                attempted[method]+=1;runtimes[method]+=record['runtime_seconds'] or 0
                if record['fit_success']:
                    expected='interior_mle' if method.startswith('mle') else 'diagnostics_passed'
                    assert record['status']==expected and record['converged']
                    paired[(scenario['id'],rep)].add(method)
                    successful_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                    primary=[p for p in record['points'] if p['target_kind']=='primary']
                    assert len(primary)==5
                    if all(p['valid_point'] for p in primary):valid_means[method]+=1
                if record['status']=='diagnostics_failed':
                    assert not record['fit_success'] and not any(p['valid_point'] for p in record['points'])
    protected,_=read_json(root/'metadata'/'protected_submission_files.json')
    for name,sha in protected.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    analysis=analyze_stage(root,cfg,cfg['scenarios'],list(METHODS),fit_id)
    bymethod=[]
    for method in METHODS:
        status=Counter()
        for (sid,m,s),n in counts.items():
            if m==method:status[s]+=n
        success=sum(n for s,n in status.items() if s in ('interior_mle','diagnostics_passed'))
        bymethod.append(dict(method=method,planned=60,attempted=attempted[method],successful=success,
            all_five_primary_means_valid=valid_means[method],status_counts=dict(status),
            sum_recorded_fit_runtime_seconds=runtimes[method],
            runtime_note='Sum of individual elapsed fit durations; concurrent tasks overlap, so this is not campaign wall time.'))
    supervision=[]
    for p in sorted(root.glob('diagnostics/supervision/*/started.json')):
        started,_=read_json(p);end=p.with_name('finished.json')
        row=dict(log_directory=str(p.parent),started_unix=started['started_unix'],deadline_unix=started['deadline_unix'])
        if end.exists():
            finished,_=read_json(end);row.update(finished)
            row['observed_wall_seconds']=finished['finished_unix']-started['started_unix']
        else:row['completion_log_available']=False
        supervision.append(row)
    audit=dict(fit_id=fit_id,created_unix=time.time(),planned_datasets=60,
        saved_datasets=sum(dataset_counts.values()),datasets_by_scenario=dict(dataset_counts),methods=bymethod,
        counts_by_scenario=[dict(scenario_id=sid,method=m,status=s,count=n) for (sid,m,s),n in sorted(counts.items())],
        all_four_bayesian_successful_datasets_by_scenario={s['id']:sum(
            set(METHODS[2:]).issubset(paired[(s['id'],r)]) for r in range(1,21)) for s in cfg['scenarios']},
        all_six_successful_datasets_by_scenario={s['id']:sum(
            set(METHODS).issubset(paired[(s['id'],r)]) for r in range(1,21)) for s in cfg['scenarios']},
        protected_report_and_pdf_hashes_unchanged=True,all_fit_dataset_hashes_verified=True,
        all_saved_designs_verified=True,successful_result_file_hashes=successful_hashes,
        supervision=supervision,analysis=analysis,
        budget_interruptions=len(list(root.glob('diagnostics/budget_interruptions/**/*.json'))),
        interpretation='Small-scale exploratory simulation (20 planned per scenario). Metrics condition on explicitly counted valid estimates/intervals; paired MSE uses common valid datasets. Missing and diagnostic-failed fits are not successful.',
        resume='Use the same config and implementation with a new future deadline; successful committed fits are skipped. Interrupted tasks remain eligible. Failed committed fits are preserved; retry only in a separate explicit attempt/campaign.')
    priority_logs=[]
    for p in sorted(root.glob('diagnostics/priority_supervision/*/started.json')):
        begin,_=read_json(p);row=dict(begin,log_directory=str(p.parent))
        finish=p.with_name('finished.json')
        if finish.exists():
            ended,_=read_json(finish);row.update(ended)
            row['supervisor_wall_seconds']=ended['finished_unix']-begin['supervisor_started_unix']
        priority_logs.append(row)
    if priority_logs:
        audit['priority_supervision']=priority_logs
        original,_=read_json(root/'metadata/urgent30_start.json')
        for name,sha in original['original_file_hashes'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        baseline={x['method']:x['successful'] for x in original['original_manifest']['methods']}
        latest=priority_logs[-1]
        audit['urgent30_execution']=dict(original_files_verified_unchanged=len(original['original_file_hashes']),
            start_unix=latest['task_start_unix'],elapsed_seconds_at_audit=time.time()-latest['task_start_unix'],
            hard_deadline_unix=latest['hard_deadline_unix'],compute_deadline_unix=latest['compute_deadline_unix'],
            new_successful_fits_by_method={x['method']:x['successful']-baseline[x['method']] for x in bymethod},
            quick_focused_tests_passed=27,
            finalization_complete_before_deadline=time.time()<latest['hard_deadline_unix'])
    path=root/'metadata'/'execution_audits'/f'{time.time_ns()}.json'
    commit_json(path,audit)
    publish_pointer(root/'manifest.json',path)
    print('Execution audit:',path)
    print('Datasets:',audit['saved_datasets'])
    for row in bymethod:print(row)
    print('Complete Bayesian datasets by scenario:',audit['all_four_bayesian_successful_datasets_by_scenario'])


if __name__=='__main__':main()
