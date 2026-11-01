"""Unexecuted tests: all estimator integration checks use mocks, never MCMC."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.stage2_example import load_example
from experiments import simulation_adapters as adapters
from experiments import simulation_runner as runner
from experiments.simulation_analysis import analyze_stage
from experiments.simulation_config import ROOT, METHODS, dataset_identity, load_config, digest
from experiments.simulation_generator import GENERATOR_VERSION, dataset_path, observed_payload, load_dataset
from experiments.simulation_storage import (CompatibilityError, TaskLocked, task_lock, commit_json,
                                           read_json, commit_bytes)
from src import bayesian
from src.bayesian_sampling import ShortestInterval
from src.censoring import BPCSExperiment, BlockData, Observation
from src.mle import MLEResult


@pytest.fixture
def cfg():
    config=load_config(ROOT/'configs/simulation_pilot.json')
    data=load_example();s=deepcopy(config['scenarios'][0]);s['id']='fixed_observed'
    for spec,b in zip(s['blocks'],data.blocks):
        spec.update(n=b.scheme.n,m=b.scheme.m,withdrawals=list(b.scheme.withdrawals))
    config.update(scenarios=[s],repetitions=1,mission_times=[])
    return config


def persist_fixture(root,cfg):
    """Serialize an existing fixed fixture, without generating observations."""
    runner.ensure_study(root,cfg,create=True)
    s=cfg['scenarios'][0];uid,seed=dataset_identity(s,1,cfg['master_seed'])
    payload=dict(schema_version=1,generator_version=GENERATOR_VERSION,dataset_id=uid,
        scenario_id=s['id'],repetition=1,scenario_definition_hash=digest(s),
        seed_information={'master_seed':cfg['master_seed'],'dataset_seed':seed},
        true_parameters={'a':s['a'],'b_by_block':{b['block']:b['b'] for b in s['blocks']}},
        number_of_blocks=2,blocks=observed_payload(load_example()),generator_implementation={})
    path=dataset_path(root,s['id'],1);commit_json(path,payload)
    return path


def test_all_six_methods_receive_same_saved_observations_and_resume(tmp_path,cfg,monkeypatch):
    path=persist_fixture(tmp_path,cfg); seen=[]
    def fake(data,scenario,config,method,seed):
        assert path.exists()
        disk,_,_=load_dataset(path)
        assert data==disk
        seen.append((id(data),method,seed))
        b=adapters._empty_bundle(method,scenario,config,seed)
        b.update(status='interior_mle' if method.startswith('mle') else 'diagnostics_passed',
                 converged=True,fit_success=True,runtime_seconds=.01)
        for p in b['points']:p.update(estimate=p['truth'],valid_point=True,point_status='valid')
        adapters._failed_intervals(b,config)
        return b,None
    monkeypatch.setattr(runner,'fit_method',fake)
    before=path.read_bytes()
    fitid,counts=runner.fit_stage(tmp_path,cfg,cfg['scenarios'],METHODS)
    assert counts['completed']==6 and len({v[0] for v in seen})==1
    assert len({v[2] for v in seen})==6
    assert path.read_bytes()==before
    second,reused=runner.fit_stage(tmp_path,cfg,cfg['scenarios'],METHODS)
    assert second==fitid and reused['reused']==6 and len(seen)==6
    for method in METHODS:
        out=runner.result_path(tmp_path,fitid,cfg['scenarios'][0]['id'],1,method)
        b,_=read_json(out)
        assert b['dataset_path']==str(path.relative_to(tmp_path))
        assert b['dataset_id']==load_dataset(path)[1]['dataset_id']
    summary=analyze_stage(tmp_path,cfg,cfg['scenarios'],list(METHODS),fitid)
    assert (tmp_path/'summaries/parameter_performance.csv').is_symlink()
    assert (tmp_path/summary['exports']['estimates/replication_estimates.csv']['path']).is_file()


def test_fit_only_never_generates_missing_data(tmp_path,cfg,monkeypatch):
    runner.ensure_study(tmp_path,cfg,create=True)
    monkeypatch.setattr(runner,'generate_and_save',lambda *a:pytest.fail('fit-only generated data'))
    monkeypatch.setattr(runner,'fit_method',lambda *a:pytest.fail('fit began before preflight'))
    with pytest.raises(CompatibilityError,match='Missing saved dataset'):
        runner.fit_stage(tmp_path,cfg,cfg['scenarios'],METHODS)


def test_changed_sampler_creates_new_campaign_not_new_data(tmp_path,cfg):
    path=persist_fixture(tmp_path,cfg);before=path.read_bytes()
    first=runner.campaign(tmp_path,cfg)
    new=deepcopy(cfg);new['mcmc']['draws']+=100
    runner.ensure_study(tmp_path,new)
    assert runner.campaign(tmp_path,new)!=first
    assert path.read_bytes()==before
    incompatible=deepcopy(cfg);incompatible['scenarios'][0]['a']=2.
    with pytest.raises(CompatibilityError):runner.ensure_study(tmp_path,incompatible)


def test_task_lock_and_immutable_result_commit(tmp_path):
    path=tmp_path/'worker.lock'
    with task_lock(path):
        with pytest.raises(TaskLocked):
            with task_lock(path):pass
    assert not path.exists()
    final=tmp_path/'result.json';commit_json(final,{'value':1.})
    with pytest.raises(CompatibilityError):commit_json(final,{'value':2.})
    assert read_json(final)[0]=={'value':1.}


def test_generation_only_never_estimates(tmp_path,cfg,monkeypatch):
    persist_fixture(tmp_path,cfg)
    monkeypatch.setattr(runner,'fit_method',lambda *a:pytest.fail('generate-only fitted'))
    monkeypatch.setattr(runner,'generate_and_save',lambda *a:pytest.fail('regenerated existing data'))
    counts=runner.generate_stage(tmp_path,cfg,cfg['scenarios'])
    assert counts['reused']==1


def test_analyze_only_reads_existing_fit_records(tmp_path,cfg,monkeypatch):
    persist_fixture(tmp_path,cfg);fitid=runner.campaign(tmp_path,cfg)
    monkeypatch.setattr(runner,'generate_stage',lambda *a:pytest.fail('analysis generated'))
    monkeypatch.setattr(runner,'fit_method',lambda *a:pytest.fail('analysis fitted'))
    first=analyze_stage(tmp_path,cfg,cfg['scenarios'],list(METHODS),fitid)
    second=analyze_stage(tmp_path,cfg,cfg['scenarios'],list(METHODS),fitid)
    assert first['analysis_id']==second['analysis_id']
    manifest,_=read_json(first['manifest'])
    assert manifest['exports']['estimates/replication_estimates.csv']['rows']>0
    text=(tmp_path/'summaries/method_success_rates.csv').read_text()
    assert 'not_run_count' in text


def fake_bayesian(cfg, *, status='diagnostics_passed', first='finite'):
    data=load_example();names=('a','b1[A]','b2[A]','b1[B]','b2[B]')
    moments={n:{'first':'finite','second':'finite'} for n in names}
    moments['b1[A]']['first']=first
    estimates={n:1. for n in names}
    if first!='finite':estimates['b1[A]']=None
    if status!='diagnostics_passed':estimates=dict.fromkeys(names)
    interval=ShortestInterval(.1,2.,.95,.95)
    diagnostics={'parameters':{n:dict(rhat=1.,ess_bulk=800.,ess_tail=700.,mcse_mean=.01) for n in names}}
    return SimpleNamespace(status=status,converged=status=='diagnostics_passed',names=names,
        estimates=estimates,intervals={n:interval for n in names},moments=moments,diagnostics=diagnostics,
        configuration={'sampler':cfg['mcmc']},warnings=[],reliability={},draws=np.ones((2,8,5)))


def test_bayesian_mean_moment_failure_does_not_discard_proper_interval(cfg,monkeypatch):
    monkeypatch.setattr(adapters.bayesian,'posterior_conditions',lambda *a,**k:{'proper':True,'status':'proper','moments':{}})
    received={}
    def fake(data,**kwargs):
        received.update(kwargs)
        return fake_bayesian(cfg,first='infinite')
    monkeypatch.setattr(adapters.bayesian,'fit',fake)
    record,draws=adapters.fit_method(load_example(),cfg['scenarios'][0],cfg,METHODS[2],123)
    row=next(p for p in record['points'] if p['parameter']=='b1[A]')
    assert not row['valid_point'] and row['estimate'] is None and row['first_moment']=='infinite'
    interval=next(i for i in record['intervals'] if i['parameter']=='b1[A]')
    assert interval['valid_interval'] and interval['interval_kind']=='credible'
    assert received['config'].seed==123 and received['prior']=='independent'
    assert draws is not None


def test_mcmc_diagnostic_failure_marks_intervals_exploratory(cfg,monkeypatch):
    monkeypatch.setattr(adapters.bayesian,'posterior_conditions',lambda *a,**k:{'proper':True,'status':'proper','moments':{}})
    monkeypatch.setattr(adapters.bayesian,'fit',lambda *a,**k:fake_bayesian(cfg,status='diagnostics_failed'))
    result,_=adapters.fit_method(load_example(),cfg['scenarios'][0],cfg,METHODS[3],44)
    assert not result['fit_success']
    assert all(not p['valid_point'] for p in result['points'])
    assert all(not i['valid_interval'] for i in result['intervals'])
    assert result['intervals'][0]['status']=='exploratory_diagnostics_failed'
    assert result['bayesian_diagnostics'][0]['rhat']==1.


@pytest.mark.parametrize('method,status',[(METHODS[2],'improper_posterior'),(METHODS[3],'unsupported_prior_data')])
def test_zero_counts_refused_without_sampling(cfg,monkeypatch,method,status):
    data=load_example()
    first=data.blocks[0]
    zero=BlockData(first.scheme,tuple(Observation(r.block,r.time,2,r.withdrawals) for r in first.observations))
    data=BPCSExperiment((zero,data.blocks[1]))
    monkeypatch.setattr(adapters.bayesian,'fit',lambda *a,**k:pytest.fail('zero-count independent posterior sampled'))
    result,_=adapters.fit_method(data,cfg['scenarios'][0],cfg,method,44)
    assert result['status']==status and not result['fit_success']


def test_numerical_and_unexpected_sampler_failures_are_records(cfg,monkeypatch):
    monkeypatch.setattr(adapters.bayesian,'posterior_conditions',lambda *a,**k:{'proper':True,'status':'proper','moments':{}})
    for error,status in [(FloatingPointError('range'),'numerical_failure'),(RuntimeError('bug'),'unexpected_error')]:
        def fail(*a,**k):raise error
        monkeypatch.setattr(adapters.bayesian,'fit',fail)
        result,_=adapters.fit_method(load_example(),cfg['scenarios'][0],cfg,METHODS[4],44)
        assert result['status']==status and result['errors']
        assert not any(p['valid_point'] for p in result['points'])


def test_classical_adapters_keep_interface_and_refuse_restricted_wald(cfg,monkeypatch):
    calls=[]
    def fake(data,**kwargs):
        calls.append((data,kwargs))
        return MLEResult(1.,{'A':(.5,1.5),'B':(.75,2.25)},0.,True,'interior_mle','profile','mock',{})
    monkeypatch.setattr(adapters,'fit_mle',fake)
    monkeypatch.setattr(adapters,'fit_order_restricted_mle',fake)
    monkeypatch.setattr(adapters,'parameter_inference',lambda *a,**k:SimpleNamespace(intervals={},diagnostics={}))
    data=load_example()
    U,_=adapters.fit_method(data,cfg['scenarios'][0],cfg,METHODS[0],1)
    R,_=adapters.fit_method(data,cfg['scenarios'][0],cfg,METHODS[1],1)
    assert all(call[0] is data for call in calls)
    assert 'constraint_tol' not in calls[0][1] and 'constraint_tol' in calls[1][1]
    assert next(i for i in R['intervals'] if i['parameter']=='b1[A]')['status']=='unsupported'
    assert all(i['interval_kind']=='confidence' for i in U['intervals'])
    from src.mle import fit_mle
    from src.order_restricted import fit_order_restricted_mle
    assert 'initial_a' in inspect.signature(fit_mle).parameters
    assert 'constraint_tol' in inspect.signature(fit_order_restricted_mle).parameters
    assert {'prior','restricted','config','times'}<=set(inspect.signature(bayesian.fit).parameters)


def test_restricted_violation_is_derived_from_truth_not_label(cfg,monkeypatch):
    s=deepcopy(cfg['scenarios'][0]);s['ordering']='arbitrary_label'
    s['blocks'][0]['b']=[1.5,.5]
    monkeypatch.setattr(adapters,'fit_mle',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('stop')))
    monkeypatch.setattr(adapters,'fit_order_restricted_mle',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('stop')))
    U,_=adapters.fit_method(load_example(),s,cfg,METHODS[0],1)
    R,_=adapters.fit_method(load_example(),s,cfg,METHODS[1],1)
    assert not U['restricted_model_misspecified'] and R['restricted_model_misspecified']


def test_dataset_result_pairing_is_verified(tmp_path,cfg):
    path=persist_fixture(tmp_path,cfg);_,meta,sha=load_dataset(path)
    target=runner.result_path(tmp_path,'campaign',cfg['scenarios'][0]['id'],1,METHODS[0])
    commit_json(target,dict(schema_version=1,fit_id='campaign',dataset_id=meta['dataset_id'],
        dataset_sha256='wrong',method=METHODS[0],scenario_id=meta['scenario_id'],repetition=1))
    with pytest.raises(CompatibilityError):runner.verify_result(target,'campaign',meta,sha,METHODS[0])
