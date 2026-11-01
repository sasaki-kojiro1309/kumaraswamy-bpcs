"""Read-only statistical aggregation of committed simulation outputs.

No sampling or fitting is invoked by analyze_stage. Exports are immutable,
versioned snapshots; top-level CSV paths are replaceable convenience symlinks.
"""
from collections import defaultdict
import csv
import io
import math
from pathlib import Path
import statistics

from .simulation_config import PAIRS, canonical, digest, fit_settings, source_hashes
from .simulation_storage import (CompatibilityError, commit_bytes, commit_json,
                                  read_json, publish_pointer, task_lock)
from .simulation_generator import dataset_path, load_dataset
from .simulation_adapters import _empty_bundle, _failed_intervals

ANALYSIS_VERSION = 'kw-bpcs-summary-v1'


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def point_metrics(rows):
    """Conditional-on-valid AB is abs(mean error), NEVER mean(abs(error))."""
    valid = [r for r in rows if r['valid_point'] and r['estimate'] is not None and math.isfinite(r['estimate'])]
    n = len(valid)
    result = dict(R_generated=len(rows), R_point=n, point_availability_rate=n/len(rows) if rows else None,
                  mean_estimate=None, signed_bias=None, absolute_bias=None, mse=None,
                  estimate_sd=None, metric_status='no_valid_estimates')
    if not n:
        return result
    try:
        errors = [r['estimate']-r['truth'] for r in valid]
        signed = _mean(errors)
        result.update(mean_estimate=_mean([r['estimate'] for r in valid]), signed_bias=signed,
                      absolute_bias=abs(signed), mse=_mean([e*e for e in errors]),
                      estimate_sd=statistics.stdev([r['estimate'] for r in valid]) if n>1 else None,
                      metric_status='conditional_on_valid_estimates')
        for key, value in list(result.items()):
            if isinstance(value, float) and not math.isfinite(value):
                result[key] = None; result['metric_status'] = 'metric_outside_float64_range'
    except (OverflowError, ValueError):
        result['metric_status'] = 'metric_outside_float64_range'
    return result


def interval_metrics(rows):
    """Empty sets count as available, covered=0, width=0. Unavailable sets do not.

    CP and AW expose separate counts if a finite-endpoint width is unrepresentable.
    Exploratory MCMC intervals are retained in exports but excluded from both.
    """
    valid = [r for r in rows if r['valid_interval'] and r['covered'] is not None]
    widths = [r['width'] for r in valid if r['width'] is not None and math.isfinite(r['width'])]
    total = len(rows)
    return dict(R_generated=total, R_CP=len(valid), R_AW=len(widths),
                coverage_probability=_mean([r['covered'] for r in valid]), average_width=_mean(widths),
                interval_availability_count=sum(bool(r['available']) for r in rows),
                valid_interval_count=len(valid), interval_availability_rate=len(valid)/total if total else None,
                empty_set_count=sum(r['status']=='empty' for r in rows),
                exploratory_count=sum(r['status']=='exploratory_diagnostics_failed' for r in rows),
                unavailable_count=sum(not r['available'] for r in rows),
                denominator='valid intervals including empty sets; widths require finite set length')


def paired_comparison(unrestricted, restricted):
    """MSEs and squared-error difference use exactly the common-valid datasets."""
    def keyed(rows):
        result = {}
        for r in rows:
            if r['valid_point'] and r['estimate'] is not None and math.isfinite(r['estimate']):
                if r['dataset_id'] in result:
                    raise CompatibilityError('duplicate estimate in paired comparison')
                result[r['dataset_id']] = r
        return result
    u, r = keyed(unrestricted), keyed(restricted)
    paired = sorted(set(u)&set(r))
    output = dict(R_U=len(u), R_R=len(r), R_paired=len(paired), MSE_U_paired=None,
                  MSE_R_paired=None, relative_mse_reduction_percent=None,
                  mean_squared_error_difference_U_minus_R=None, paired_difference_mcse=None,
                  success_sets_equal=set(u)==set(r), comparison_status='no_common_valid_estimates',
                  comparison_subset='common_valid_dataset_ids')
    if not paired:
        return output
    for uid in paired:
        if u[uid]['truth'] != r[uid]['truth'] or u[uid]['dataset_sha256'] != r[uid]['dataset_sha256']:
            raise CompatibilityError('paired estimates disagree on data or truth')
    try:
        eu = [(u[d]['estimate']-u[d]['truth'])**2 for d in paired]
        er = [(r[d]['estimate']-r[d]['truth'])**2 for d in paired]
        differences = [x-y for x,y in zip(eu,er)]
        mu, mr = _mean(eu), _mean(er)
        output.update(MSE_U_paired=mu, MSE_R_paired=mr,
                      relative_mse_reduction_percent=100*(mu-mr)/mu if mu>0 else None,
                      mean_squared_error_difference_U_minus_R=_mean(differences),
                      paired_difference_mcse=statistics.stdev(differences)/math.sqrt(len(paired)) if len(paired)>1 else None,
                      comparison_status='computed' if mu>0 else 'zero_unrestricted_MSE_ratio_undefined')
        for key,value in list(output.items()):
            if isinstance(value,float) and not math.isfinite(value):
                output[key]=None; output['comparison_status']='metric_outside_float64_range'
    except (OverflowError, ValueError):
        output['comparison_status']='metric_outside_float64_range'
    return output


def _csv(rows):
    stream = io.StringIO(newline='')
    first = ['analysis_id','fit_id','scenario_id','ordering','censoring','method','dataset_id',
             'repetition','parameter','target_kind','truth','restricted_model_misspecified']
    fields = set().union(*(r.keys() for r in rows)) if rows else set(first)
    columns = [k for k in first if k in fields]+sorted(fields-set(first))
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in sorted(rows, key=lambda r: tuple(str(r.get(k,'')) for k in first)):
        writer.writerow({k: canonical(v).decode() if isinstance(v,(dict,list,tuple)) else v
                         for k,v in row.items()})
    return stream.getvalue().encode('utf-8')


def _groups(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    return grouped


def _scenario_columns(scenario):
    return dict(ordering=scenario['ordering'], censoring=scenario['censoring'], true_a=scenario['a'],
                block_design=[{key: b[key] for key in ('block','n','m','withdrawals','b')} for b in scenario['blocks']])


def analyze_stage(root, cfg, scenarios, methods, fit_id):
    """Use one recorded fit campaign. Never combine settings/environment IDs."""
    from .simulation_runner import ensure_study, result_path, verify_result, generation_failure_path
    root = Path(root)
    ensure_study(root, cfg)
    saved_campaign, _ = read_json(root/'metadata'/'fit_configs'/f'{fit_id}.json')
    if saved_campaign.get('fit_id') != fit_id:
        raise CompatibilityError('invalid campaign metadata')
    # Use recorded settings rather than the analysis machine's sampler defaults.
    analysis_cfg = dict(cfg)
    analysis_cfg.update(saved_campaign['settings'])
    points, intervals, diagnostics, failures, runtimes, bundles, generation = [], [], [], [], [], [], []
    dependencies, scenario_map = [], {s['id']:s for s in scenarios}
    for scenario in scenarios:
        for rep in range(1,cfg['repetitions']+1):
            path = dataset_path(root, scenario['id'], rep)
            if not path.exists():
                failure = generation_failure_path(root, scenario['id'], rep)
                row = dict(scenario_id=scenario['id'], repetition=rep, status='not_generated')
                if failure.exists():
                    item, sha = read_json(failure); dependencies.append(('generation_failure',sha))
                    row.update(item)
                generation.append(row)
                continue
            _, metadata, data_sha = load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
            dependencies.append(('dataset', data_sha))
            generation.append(dict(scenario_id=scenario['id'], repetition=rep, status='generated', dataset_id=metadata['dataset_id']))
            for method in methods:
                task = result_path(root,fit_id,scenario['id'],rep,method)
                if task.exists():
                    b, task_sha = verify_result(task,fit_id,metadata,data_sha,method)
                    dependencies.append(('result',task_sha))
                else:
                    b = _empty_bundle(method,scenario,analysis_cfg,None)
                    b.update(status='not_run', scenario_id=scenario['id'], repetition=rep,
                             dataset_id=metadata['dataset_id'], dataset_sha256=data_sha, fit_id=fit_id,
                             ordering=scenario['ordering'],censoring=scenario['censoring'])
                    _failed_intervals(b,analysis_cfg)
                # Reject inconsistent repeated IDs rather than silently merging.
                context = {key:b.get(key) for key in ('fit_id','dataset_id','dataset_sha256','scenario_id',
                    'repetition','method','restricted','restricted_model_misspecified','ordering','censoring')}
                context['fit_status'] = b['status']
                context['posterior_propriety_status'] = b['posterior_propriety_status']
                bundles.append(b)
                for point in b['points']:
                    points.append(dict(**context, **point))
                kinds = {p['parameter']:p['target_kind'] for p in b['points']}
                truths = {p['parameter']:p['truth'] for p in b['points']}
                for interval in b['intervals']:
                    intervals.append(dict(**context, **interval, target_kind=kinds[interval['parameter']],
                                          truth=truths[interval['parameter']]))
                runtimes.append(dict(**context,runtime_seconds=b['runtime_seconds'],
                    fit_success=b['fit_success'],converged=b['converged'],
                    attained_mle=b['attained_mle'],boundary_supremum=b['boundary_supremum']))
                for diag in b['bayesian_diagnostics']:
                    diagnostics.append(dict(**context,**diag,mcmc_seed=b['mcmc_seed'],**(b['mcmc'] or {})))
                if b['status'] not in ('interior_mle','diagnostics_passed'):
                    failures.append(dict(**context,stage='fit_status',message=b['status']))
                for error in b['errors']:
                    failures.append(dict(**context,**error))
                for point in b['points']:
                    if point['first_moment'] not in ('finite','not_applicable'):
                        failures.append(dict(**context,stage='posterior_moment',parameter=point['parameter'],
                                             message=point['point_status']))
    analysis_definition=dict(version=ANALYSIS_VERSION, fit_id=fit_id, methods=sorted(methods),
        scenarios=sorted(scenario_map), repetitions=cfg['repetitions'], dependencies=sorted(dependencies),
        implementation=source_hashes(('experiments/simulation_analysis.py','experiments/simulation_adapters.py',
                                      'experiments/simulation_storage.py')))
    analysis_id=digest(analysis_definition)[:24]
    common=dict(analysis_id=analysis_id,fit_id=fit_id)
    performance=[]
    for key,rows in _groups(points,('scenario_id','method','parameter','target_kind')).items():
        sid,method,name,kind=key
        if len({r['dataset_id'] for r in rows}) != len(rows) or len({r['truth'] for r in rows}) != 1:
            raise CompatibilityError('duplicate/mixed truth in parameter performance group')
        performance.append(dict(**common,scenario_id=sid,method=method,parameter=name,target_kind=kind,
            truth=rows[0]['truth'],restricted_model_misspecified=rows[0]['restricted_model_misspecified'],
            **_scenario_columns(scenario_map[sid]),**point_metrics(rows)))
    interval_performance=[]
    for key,rows in _groups(intervals,('scenario_id','method','parameter','interval_family','interval_kind','nominal_probability')).items():
        sid,method,name,family,kind,nominal=key
        if len({r['dataset_id'] for r in rows}) != len(rows):
            raise CompatibilityError('duplicate intervals in a performance group')
        interval_performance.append(dict(**common,scenario_id=sid,method=method,parameter=name,
            target_kind=rows[0]['target_kind'],interval_family=family,interval_kind=kind,nominal_probability=nominal,
            restricted_model_misspecified=rows[0]['restricted_model_misspecified'],
            **_scenario_columns(scenario_map[sid]),**interval_metrics(rows)))
    rates=[];pooling=[]
    for (sid,method),rows in _groups(bundles,('scenario_id','method')).items():
        n=len(rows); good=sum(b['fit_success'] for b in rows)
        def count(status): return sum(b['status']==status for b in rows)
        timed=[b['runtime_seconds'] for b in rows if b['runtime_seconds'] is not None]
        valid_counts=[sum(p['valid_point'] for p in b['points'] if p['target_kind']=='primary') for b in rows]
        primary_count=1+2*len(scenario_map[sid]['blocks'])
        rates.append(dict(**common,scenario_id=sid,method=method,**_scenario_columns(scenario_map[sid]),
            R_generated=n,R_attempted=sum(b['status']!='not_run' for b in rows),
            successful_fit_count=good,successful_fit_rate=good/n,
            all_primary_estimates_valid_count=sum(c==primary_count for c in valid_counts),
            all_primary_estimates_valid_rate=sum(c==primary_count for c in valid_counts)/n,
            attained_mle_count=sum(b['attained_mle'] for b in rows),boundary_supremum_count=count('boundary_supremum'),
            mathematical_nonexistence_count=count('unbounded_likelihood'),not_run_count=count('not_run'),
            numerical_failure_rate=count('numerical_failure')/n,
            unexpected_error_rate=count('unexpected_error')/n,
            optimization_failure_rate=count('optimization_failed')/n,
            posterior_impropriety_rate=count('improper_posterior')/n,
            unsupported_prior_data_rate=count('unsupported_prior_data')/n,
            posterior_mean_infinite_or_unverified_rate=sum(b['status']!='not_run' and any(p['first_moment'] in ('infinite','unverified','undefined')
                for p in b['points'] if p['target_kind']=='primary') for b in rows)/n,
            bayesian_convergence_failure_rate=count('diagnostics_failed')/n,
            inference_error_rate=sum(bool(b['errors']) for b in rows)/n,
            timed_fit_count=len(timed),average_computation_seconds=_mean(timed),
            average_successful_computation_seconds=_mean([b['runtime_seconds'] for b in rows if b['fit_success']]),
            restricted_model_misspecified=rows[0]['restricted_model_misspecified']))
        if method=='mle_within_block':
            verified=[b for b in rows if b['converged'] and b['status'] in ('interior_mle','boundary_supremum')]
            for block in scenario_map[sid]['blocks']:
                label=block['block']; denominator=len(verified)
                active=sum(bool(b['active_constraints'].get(label)) for b in verified)
                pooled=sum(bool(b['pooled_blocks'].get(label)) for b in verified)
                pooling.append(dict(**common,scenario_id=sid,method=method,block=label,R_generated=n,
                    R_verified_order_fits=denominator,active_equality_count=active,strict_pooling_count=pooled,
                    active_equality_frequency=active/denominator if denominator else None,
                    strict_pooling_frequency=pooled/denominator if denominator else None))
    comparisons=[]
    groups=_groups(points,('scenario_id','method','parameter','target_kind'))
    for sid in scenario_map:
        names=sorted({(p['parameter'],p['target_kind']) for p in points if p['scenario_id']==sid})
        for u,r in PAIRS:
            if u not in methods or r not in methods: continue
            for name,kind in names:
                comparisons.append(dict(**common,scenario_id=sid,parameter=name,target_kind=kind,
                    unrestricted_method=u,restricted_method=r,**_scenario_columns(scenario_map[sid]),
                    restricted_model_misspecified=any(b['b'][0]>b['b'][1] for b in scenario_map[sid]['blocks']),
                    **paired_comparison(groups.get((sid,u,name,kind),[]),groups.get((sid,r,name,kind),[]))))
    mcmc=[]
    for (sid,method,name),rows in _groups(diagnostics,('scenario_id','method','parameter')).items():
        def values(key): return [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        rhats,bulk,tail=values('rhat'),values('ess_bulk'),values('ess_tail')
        mcmc.append(dict(**common,scenario_id=sid,method=method,parameter=name,R_generated=len(rows),
            R_finite_rhat=len(rhats),max_rhat=max(rhats) if rhats else None,mean_rhat=_mean(rhats),
            minimum_bulk_ess=min(bulk) if bulk else None,minimum_tail_ess=min(tail) if tail else None,
            mcse_available_count=len(values('mcse_mean')),mean_reported_mcse=_mean(values('mcse_mean'))))
    generation_summary=[]
    for (sid,),rows in _groups(generation,('scenario_id',)).items():
        generation_summary.append(dict(**common,scenario_id=sid,R_requested=len(rows),
            R_generated=sum(r['status']=='generated' for r in rows),
            R_generation_failed=sum(r['status']=='generation_failure' for r in rows),
            R_not_generated=sum(r['status']=='not_generated' for r in rows)))
    tables={
        'estimates/replication_estimates.csv':points,
        'estimates/replication_intervals.csv':intervals,
        'diagnostics/bayesian_diagnostics.csv':diagnostics,
        'diagnostics/failure_log.csv':failures+[dict(stage='generation',**r) for r in generation if r['status']!='generated'],
        'diagnostics/runtime.csv':runtimes,
        'summaries/parameter_performance.csv':performance,
        'summaries/interval_performance.csv':interval_performance,
        'summaries/method_success_rates.csv':rates,
        'summaries/relative_mse_comparison.csv':comparisons,
        'summaries/mcmc_summary.csv':mcmc,
        'summaries/active_pooling.csv':pooling,
        'summaries/generation_counts.csv':generation_summary,
    }
    manifest_path=root/'metadata'/'analyses'/f'{analysis_id}.json'
    with task_lock(root/'metadata'/'locks'/'analysis.lock'):
        exports={}
        for relative,rows in tables.items():
            for row in rows: row.update(common)
            category,filename=relative.split('/')
            versioned=root/category/'exports'/fit_id/analysis_id/filename
            sha=commit_bytes(versioned,_csv(rows))
            exports[relative]=dict(path=str(versioned.relative_to(root)),sha256=sha,rows=len(rows))
        commit_json(manifest_path,dict(analysis_id=analysis_id,definition=analysis_definition,exports=exports,
            note='Conditional on stated valid observations; no universal-best-estimator rule.'))
        # Pointers are published only after every versioned artifact is committed.
        # The manifest is the transactional source of truth if interrupted here.
        for relative,item in exports.items():
            publish_pointer(root/relative,root/item['path'])
        publish_pointer(root/'metadata'/'latest_analysis.json',manifest_path)
    return {'analysis_id':analysis_id,'manifest':str(manifest_path),'exports':exports}
