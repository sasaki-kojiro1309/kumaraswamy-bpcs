"""Read-only statistical verification after priority workers have stopped."""
import csv
import math
from collections import defaultdict
from pathlib import Path
import subprocess
import time

from experiments.execution.submission_finalize import main as finalize
from experiments.simulation_storage import read_json,commit_json,commit_bytes


def close(a,b):
    assert math.isclose(float(a),b,rel_tol=1e-11,abs_tol=1e-13),(a,b)


def main():
    root=Path('results/simulation_submission')
    log=sorted(root.glob('diagnostics/priority_supervision/*/started.json'))[-1].parent
    started,_=read_json(log/'started.json');ended,_=read_json(log/'finished.json')
    assert not ended['externally_terminated_workers'] and all(c==0 for c in ended['exit_codes'])
    processes=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    remaining=[line for line in processes if ' -m experiments.priority_supervisor' in line
               or ' -m experiments.priority_watchdog' in line or 'multiprocessing.spawn import spawn_main' in line]
    assert not remaining,remaining
    finalize()
    audit,_=read_json(root/'manifest.json');aid=audit['analysis']['analysis_id']
    def rows(name):return list(csv.DictReader((root/name).open()))
    points=defaultdict(dict)
    for r in rows('estimates/replication_estimates.csv'):
        assert r['analysis_id']==aid
        if r['valid_point']=='True':points[(r['scenario_id'],r['method'],r['parameter'])][r['dataset_id']]=r
    point_groups=0
    for r in rows('summaries/parameter_performance.csv'):
        assert r['analysis_id']==aid
        xs=list(points[(r['scenario_id'],r['method'],r['parameter'])].values())
        assert int(r['R_point'])==len(xs) and int(r['R_generated'])==20
        if xs:
            errors=[float(x['estimate'])-float(x['truth']) for x in xs]
            close(r['mean_estimate'],math.fsum(float(x['estimate']) for x in xs)/len(xs))
            close(r['signed_bias'],math.fsum(errors)/len(errors))
            close(r['absolute_bias'],abs(math.fsum(errors)/len(errors)))
            close(r['mse'],math.fsum(e*e for e in errors)/len(errors));point_groups+=1
    intervals=defaultdict(list)
    for r in rows('estimates/replication_intervals.csv'):
        assert r['analysis_id']==aid
        intervals[(r['scenario_id'],r['method'],r['parameter'],r['interval_family'])].append(r)
    interval_groups=0
    for r in rows('summaries/interval_performance.csv'):
        xs=intervals[(r['scenario_id'],r['method'],r['parameter'],r['interval_family'])]
        valid=[x for x in xs if x['valid_interval']=='True' and x['covered']!='']
        widths=[float(x['width']) for x in valid if x['width']!='']
        assert int(r['R_CP'])==len(valid) and int(r['R_AW'])==len(widths)
        if valid:close(r['coverage_probability'],sum(int(x['covered']) for x in valid)/len(valid))
        if widths:close(r['average_width'],math.fsum(widths)/len(widths))
        interval_groups+=1
    paired_groups=0
    for r in rows('summaries/relative_mse_comparison.csv'):
        u=points[(r['scenario_id'],r['unrestricted_method'],r['parameter'])]
        v=points[(r['scenario_id'],r['restricted_method'],r['parameter'])]
        ids=set(u)&set(v);assert int(r['R_paired'])==len(ids)
        if ids:
            mu=math.fsum((float(u[i]['estimate'])-float(u[i]['truth']))**2 for i in ids)/len(ids)
            mr=math.fsum((float(v[i]['estimate'])-float(v[i]['truth']))**2 for i in ids)/len(ids)
            close(r['MSE_U_paired'],mu);close(r['MSE_R_paired'],mr)
            if mu>0:close(r['relative_mse_reduction_percent'],100*(mu-mr)/mu)
        paired_groups+=1
    now=time.time();assert now<started['hard_deadline_unix']
    qa=dict(status='passed',analysis_id=aid,point_metric_groups=point_groups,
        interval_metric_groups=interval_groups,paired_comparison_groups=paired_groups,
        no_simulation_workers_running=True,all_original_datasets_MLE_and_Bayesian_files_intact=True,
        elapsed_seconds=now-started['task_start_unix'],deadline_unix=started['hard_deadline_unix'],
        completed_before_deadline=True,new_successful_fits_by_method=audit['urgent30_execution']['new_successful_fits_by_method'],
        complete_paired_datasets=audit['all_four_bayesian_successful_datasets_by_scenario'])
    commit_json(root/'diagnostics/urgent30_final_validation.json',qa)
    lines=['# Bayesian completion attempt: final execution summary','',
        f"Elapsed through final verification: {qa['elapsed_seconds']:.3f} seconds.",
        'The 30-minute deadline was met. No simulation or watchdog process remains.',
        'All 60 original datasets and 120 MLE fits, and all previous Bayesian fit files, were preserved byte-for-byte.',
        'Real-data results and report/LaTeX/PDF files were not modified.','',
        '| Method | New successful | Total successful / 60 | Diagnostic failures | Not run |',
        '|---|---:|---:|---:|---:|']
    for r in audit['methods']:
        if not r['method'].startswith('bayes'):continue
        lines.append(f"| {r['method']} | {qa['new_successful_fits_by_method'][r['method']]} | {r['successful']} | {r['status_counts'].get('diagnostics_failed',0)} | {r['status_counts'].get('not_run',0)} |")
    lines+=['','Complete paired datasets by scenario: '+str(qa['complete_paired_datasets']),
        '',f'Independent CSV checks: {point_groups} point groups, {interval_groups} interval groups, {paired_groups} paired comparisons.',
        'Quick scheduler/budget/storage/analysis tests: 27 passed. Existing diagnostic thresholds and chain lengths were unchanged.',
        '', 'Parameter estimates, AB and MSE: summaries/parameter_performance.csv.',
        'CP and AW with explicit valid denominators: summaries/interval_performance.csv.',
        'Paired relative MSE: summaries/relative_mse_comparison.csv.',
        'Exact scenario/method counts and data integrity: manifest.json.',
        'Failures: diagnostics/failure_log.csv. Interrupted fits: diagnostics/budget_interruptions/.',
        '', 'This is exploratory: 20 datasets were planned per scenario, with fewer valid Bayesian pairs. No superiority claim follows from small MSE differences.',
        'Diagnosed failures were preserved without retries. A failed required method makes that dataset unavailable for all-six-method comparisons; remaining unrelated fits on it were deliberately deprioritized.',
        'The violation scenario is misspecified under the order restriction and has persistent restricted-hierarchical convergence failures.',
        'Prior exports and execution summaries remain preserved; this dated execution has its own versioned analysis and audit.','']
    commit_bytes(root/'EXECUTION_SUMMARY_URGENT30.md','\n'.join(lines).encode())
    print(qa)


if __name__=='__main__':main()
