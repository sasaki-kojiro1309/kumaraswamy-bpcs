"""Written for later execution by the user. NOT executed during implementation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.simulation_config import (ROOT, METHODS, load_config, normalize_config,
    generation_definition, dataset_identity, seed_for, withdrawal_schedule, ConfigurationError)
from experiments.simulation_generator import (GenerationError, inverse_kumaraswamy,
    apply_progressive_censoring, generate_experiment, make_dataset, generate_and_save,
    load_dataset, dataset_path)
from experiments.simulation_storage import CompatibilityError, commit_json, read_json
from src.censoring import BPCSExperiment, BlockScheme
from src.model import Kumaraswamy


@pytest.fixture
def config():
    return load_config(ROOT/'configs/simulation_pilot.json')


@pytest.fixture
def scenario(config):
    s=deepcopy(config['scenarios'][0])
    s['id']='small_early'
    for b in s['blocks']:
        b.update(n=8,m=5,withdrawals=[3,0,0,0,0])
    return s


@pytest.mark.parametrize('a,b',[(1.,1.),(.8,2.),(3.,.5)])
def test_inverse_cdf_identity(a,b):
    u=np.array([.001,.1,.3,.7,.99])
    t=inverse_kumaraswamy(u,a,b)
    np.testing.assert_allclose(Kumaraswamy(a,b).cdf(t),u,rtol=2e-12,atol=1e-14)
    assert np.all(np.diff(t)>0)


@pytest.mark.parametrize('bad',[[0.],[1.],[-.1],[np.nan]])
def test_inverse_rejects_endpoints_without_clipping(bad):
    with pytest.raises(ValueError): inverse_kumaraswamy(bad,1.,1.)


def test_inverse_range_failure_is_not_repaired():
    with pytest.raises(GenerationError): inverse_kumaraswamy([.5],1e300,1.)


@pytest.mark.parametrize('plan', ['early','late','distributed'])
def test_schedules_and_risk_accounting(plan):
    R=withdrawal_schedule(50,30,plan)
    scheme=BlockScheme('A',50,30,R)
    assert len(R)==30 and sum(R)==20
    assert scheme.risk_sets[-1]-1-R[-1]==0
    if plan=='distributed':
        assert max(R)-min(R)<=1
        assert all(abs(sum(R[:j])-j*20/30)<1 for j in range(31))
    if plan=='early': assert R==(20,)+(0,)*29
    if plan=='late': assert R==(0,)*29+(20,)


class FirstSurvivorRemoval:
    def __init__(self): self.calls=[]
    def choice(self,active,size,replace):
        self.calls.append((active.copy(),size,replace))
        assert replace is False
        return active[:size]


def test_withdrawn_early_failure_is_never_observed():
    scheme=BlockScheme('A',4,2,(1,1)); rng=FirstSurvivorRemoval()
    b=apply_progressive_censoring(scheme,[.1,.2,.3,.4],[2,1,2,1],rng)
    assert [r.time for r in b.observations]==[.1,.3]
    assert [r.cause for r in b.observations]==[2,2]
    np.testing.assert_array_equal(rng.calls[0][0],[1,2,3])
    assert [c[1] for c in rng.calls]==[1,1]
    assert [r.withdrawals for r in b.observations]==[1,1]


def test_survivor_selection_is_by_unit_index_not_lifetime_or_cause():
    class Choice:
        def __init__(self): self.sizes=[]
        def choice(self,active,size,replace):
            self.sizes.append(len(active))
            # Choice of unit 3 skips a later latent failure, without looking at times.
            return np.array([3]) if len(active)==3 else active[:size]
    choice=Choice()
    b=apply_progressive_censoring(BlockScheme('A',4,2,(1,1)),[.1,.2,.3,.4],[1,2,1,2],choice)
    assert [r.time for r in b.observations]==[.1,.2]
    assert choice.sizes==[3,1]


def test_withdrawal_choice_frequencies_small_fixed_seed_check():
    # A small sampler-component check, not a performance simulation study.
    observed=np.zeros(3,dtype=int)
    for seed in range(240):
        class RecordChoice:
            def choice(self,active,size,replace):
                draw=np.random.default_rng(seed).choice(active,size=size,replace=replace)
                if len(active)==3:
                    for j in draw: observed[j-1]+=1
                return draw
        apply_progressive_censoring(BlockScheme('A',4,2,(1,1)),[.1,.2,.3,.4],[1,1,2,2],RecordChoice())
    # Broad deterministic component check: always choosing a sorted endpoint
    # would fail. This is not a test of estimator performance.
    assert observed.sum()==240 and np.all(np.abs(observed-80)<35)


def test_cause_labels_from_controlled_independent_latent_draws(monkeypatch,scenario):
    import experiments.simulation_generator as module
    arrays=iter([np.array([.1,.4,.3,.7,.6,.8,.85,.9]),
                 np.array([.2,.2,.5,.5,.7,.9,.95,.99])]*2)
    monkeypatch.setattr(module,'inverse_kumaraswamy',lambda *a:next(arrays))
    for b in scenario['blocks']: b.update(n=8,m=8,withdrawals=[0]*8)
    data=generate_experiment(scenario,17)
    assert [r.cause for r in data.blocks[0].observations]==[1,2,1,2,1,1,1,1]


@pytest.mark.parametrize('plan',['early','late','distributed'])
def test_generated_data_valid_ordered_and_reproducible(scenario,plan):
    for b in scenario['blocks']: b['withdrawals']=list(withdrawal_schedule(8,5,plan))
    first,second=generate_experiment(scenario,234),generate_experiment(scenario,234)
    assert isinstance(first,BPCSExperiment) and first==second
    for b in first.blocks:
        assert all(x.time<y.time for x,y in zip(b.observations,b.observations[1:]))
        assert b.scheme.n==b.scheme.m+sum(b.scheme.withdrawals)
        assert len(b.observations)==5


def test_save_reload_and_no_latent_fields(tmp_path,scenario):
    path=generate_and_save(tmp_path,scenario,1,11,{'test':'fixture'})
    data,payload,sha=load_dataset(path,scenario=scenario,repetition=1,master_seed=11)
    assert payload['number_of_blocks']==2 and payload['true_parameters']['a']==1.5
    assert len(sha)==64
    assert data==generate_experiment(scenario,payload['seed_information']['dataset_seed'])
    raw=json.loads(path.read_text()); assert set(raw)=={'payload','sha256'}
    for block in payload['blocks']:
        for row in block['observations']:
            assert set(row)=={'block','time','cause','withdrawals'}
    original=path.read_bytes()
    assert generate_and_save(tmp_path,scenario,1,11,{'test':'fixture'})==path
    assert path.read_bytes()==original


def test_corruption_and_no_overwrite(tmp_path,scenario):
    path=generate_and_save(tmp_path,scenario,1,11,{})
    original=path.read_bytes()
    with pytest.raises(CompatibilityError): commit_json(path,{'different':'payload'})
    assert path.read_bytes()==original
    path.chmod(0o644); path.write_bytes(original.replace(b'0.5',b'0.6',1))
    # Force a corruption even if that particular textual number was absent.
    if path.read_bytes()==original: path.write_text('{"payload":{},"sha256":"wrong"}')
    with pytest.raises(CompatibilityError): load_dataset(path)


def test_seed_domains_and_budget_independence(config,scenario):
    cfg2=deepcopy(config);cfg2['mcmc']['draws']*=2;cfg2['mcmc_master_seed']+=1
    cfg2['methods']=[METHODS[0]];cfg2['repetitions']=1
    assert generation_definition(config)==generation_definition(cfg2)
    uid,seed=dataset_identity(scenario,1,config['master_seed'])
    assert seed!=seed_for(config['mcmc_master_seed'],'kw-bpcs-mcmc-v1',uid,METHODS[2])
    assert dataset_identity(scenario,1,config['master_seed'])==dataset_identity(scenario,1,cfg2['master_seed'])
    assert seed!=dataset_identity(scenario,2,config['master_seed'])[1]


def test_explicit_configurations_and_invalid_schemes(config):
    main=load_config(ROOT/'configs/simulation_main.json')
    assert config['repetitions']==50 and main['repetitions']==2000
    assert len(config['scenarios'])==len(main['scenarios'])==12
    assert config['methods']==list(METHODS)
    for s in config['scenarios']:
        assert [sum(b['b']) for b in s['blocks']]==[2.,3.]
    broken=deepcopy(config);broken['scenarios'][0]['blocks'][0]['withdrawals']=[0]*30
    with pytest.raises(ValueError): normalize_config(broken)
    broken=deepcopy(config);broken['mcmc']['seed']=1
    with pytest.raises(ConfigurationError): normalize_config(broken)
