"""One-block, complete-data sensitivity application of all six existing APIs.

Source: Shi, Yan & Peng (2025), Table 6, reproducing Nelson (1982).
No simulated truth, inferred withdrawals, jitter, or removed records are used.
"""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from src import bayesian
from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.mle import fit_mle
from src.order_restricted import fit_order_restricted_mle
from src.model import Kumaraswamy
from src.uncertainty import parameter_inference, reliability_inference
from .simulation_adapters import json_safe
from .simulation_budget import ExecutionBudget, BudgetExpired
from .simulation_config import load_config, hierarchical_prior, METHODS, digest
from .simulation_generator import observed_payload
from .simulation_storage import commit_json, read_json, save_draws

ROOT = Path(__file__).resolve().parents[1]/'results'/'real_data'


def prepare():
    source = ROOT/'original'/'appliance_article.xml'
    tree = ET.parse(source).getroot()
    table = next(t for t in tree.iter('table-wrap') if t.findtext('label') == 'Table 6')
    rows, native_cause = [], None
    for tr in table.findall('./table/tbody/tr'):
        cells = tr.findall('td')
        if len(cells) == 2:
            label = ''.join(cells[0].itertext())
            native_cause = 1 if '1' in label else 0
        for text in ''.join(cells[-1].itertext()).split(','):
            if text.strip():
                rows.append(dict(source_table='Table 6', source_value_1000_cycles=text.strip(),
                    source_cause=native_cause, cause=1 if native_cause == 1 else 2,
                    time=float(text.strip())/5.0))
    assert len(rows)==39 and sum(r['cause']==1 for r in rows)==26
    rows.sort(key=lambda r:r['time'])
    assert len({r['time'] for r in rows})==39
    scheme = BlockScheme('appliances',39,39,(0,)*39)
    data = BPCSExperiment((BlockData(scheme,tuple(Observation('appliances',r['time'],r['cause'],0) for r in rows)),))
    metadata = dict(dataset='Nelson appliance data via Shi et al. (2025), Table 6',
        source_url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11791261/fullTextXML',
        article_doi='10.1016/j.heliyon.2025.e41784', original_reference='Nelson W. (1982), Applied Life Data Analysis',
        original_book_independently_accessed=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        observations=39,blocks=1,cause_counts={'1':26,'2':13},
        cause_meaning={'1':'failure mechanism 5','2':'all other failure mechanisms (source delta=0)'},
        censoring='Published table records a failure cause for every one of N=39 devices; no censoring indicated.',
        design='Complete-data special case: n=m=39, all actual withdrawals zero; one cohort, no artificial blocks.',
        transformation='time = source_value_1000_cycles / 5; fixed chosen analysis scale of 5000 cycles, maximum recorded 4129 cycles.',
        limitations=['5000 cycles is an analysis scale, not a verified physical upper bound; fit is conditional on this choice.',
            'Published transcription was verified against article XML, not original laboratory records.',
            'Source investigates dependence of mechanisms. Independent latent risks and a common shape are working assumptions, not established facts.',
            'Recorded precision is treated as exact for this exploratory application; no tied recorded times exist.',
            'One cohort validates the complete-data special case only, not an actual progressive-withdrawal experiment.'])
    commit_json(ROOT/'appliance_source_metadata.json',metadata)
    payload=dict(dataset='appliances',source_sha256=metadata['source_sha256'],source_rows=rows,blocks=observed_payload(data))
    sha=commit_json(ROOT/'processed'/'appliances.json',payload)
    # Reload persisted observations before any estimator sees the data.
    saved,_=read_json(ROOT/'processed'/'appliances.json')
    bs=saved['blocks'][0]
    reloaded=BPCSExperiment((BlockData(BlockScheme(**bs['scheme']),tuple(Observation(**r) for r in bs['observations'])),))
    commit_json(ROOT/'validation'/'appliance_compatibility.json',dict(structurally_compatible=True,
        source_observations=39,fitted_block_count=1,all_times_distinct=True,no_missing_causes=True,
        withdrawals_zero_supported_by_source=True,scientific_model_validated=False,
        observed_cause_1_fraction=26/39,order_restriction_empirically_supported=False,
        note='Observed cause 1 is more common; b1<=b2 implies p1<=0.5, so restricted MLE pools the cause parameters.'))
    return reloaded, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget-seconds',type=float,default=900)
    args=parser.parse_args();started=time.monotonic();budget=ExecutionBudget(args.budget_seconds,reserve_seconds=5)
    cfg=load_config(Path('configs/simulation_submission.json'));data,sha=prepare()
    settings_id=digest(dict(mcmc=cfg['mcmc'],prior=cfg['hierarchical_prior'],dataset_sha256=sha))[:16]
    outcomes=[]
    for index,method in enumerate(METHODS):
        path=ROOT/'estimates'/f'appliances_{settings_id}_{method}.json'
        if path.exists():
            record,_=read_json(path);outcomes.append(dict(method=method,status=record['status'],reused=True));continue
        if budget.remaining()<=0:break
        t=time.monotonic();restricted=method.endswith('within_block')
        record=dict(method=method,dataset_sha256=sha,dataset_path=str(ROOT/'processed'/'appliances.json'),
                    truth_known=False,restricted=restricted,settings_id=settings_id)
        try:
            with budget.supervise():
                if method.startswith('mle'):
                    fit=(fit_order_restricted_mle if restricted else fit_mle)(data)
                    record.update(status=fit.status,fit_success=fit.converged and fit.status=='interior_mle',result=json_safe(fit))
                    if record['fit_success']:
                        intervals=json_safe(parameter_inference(data,fit,restricted=restricted))
                        record['reliability']={str(t):json_safe(reliability_inference(data,fit,t,restricted=restricted)) for t in cfg['mission_times']}
                        b1,b2=fit.b_by_block['appliances'];model=Kumaraswamy(fit.a,b1+b2)
                        observed=np.array([r.time for r in data.blocks[0].observations]);F=np.asarray(model.cdf(observed));n=len(observed)
                        ks=float(max(np.max(np.arange(1,n+1)/n-F),np.max(F-np.arange(n)/n)))
                        grid=np.linspace(.01,.99,99)
                        validation=dict(KS_statistic=ks,KS_pvalue=None,
                            note='Descriptive fitted-CDF discrepancy; no calibrated p-value with estimated parameters. No proof of model adequacy.',
                            observed_cause_1_fraction=26/39,fitted_cause_1_fraction=b1/(b1+b2),
                            empirical_survival=[dict(time=float(x),survival_after=(n-i-1)/n) for i,x in enumerate(observed)],
                            fitted_curve=[dict(time=float(x),survival=float(model.sf(x)),hazard=float(model.hazard(x)),
                                cause_1_incidence=float(b1/(b1+b2)*model.cdf(x)),
                                cause_2_incidence=float(b2/(b1+b2)*model.cdf(x))) for x in grid])
                        commit_json(ROOT/'validation'/f'appliances_{settings_id}_{method}.json',validation)
                    else:intervals=dict(status='unavailable',reason=fit.status)
                else:
                    prior='independent' if method.startswith('bayes_independent') else hierarchical_prior(cfg['hierarchical_prior'])
                    seed=20261110+index
                    fit=bayesian.fit(data,prior=prior,restricted=restricted,
                        config=bayesian.SamplingConfig(**cfg['mcmc'],seed=seed),times=cfg['mission_times'])
                    result=asdict(fit);draws=result.pop('draws')
                    for entries in result['reliability'].values():
                        for entry in entries.values():entry.pop('draws',None)
                    record.update(status=fit.status,fit_success=fit.converged and fit.status=='diagnostics_passed',
                        result=json_safe(result),mcmc_seed=seed)
                    if draws.size:
                        drawdir=ROOT/'estimates'/'draws'/f'appliances_{settings_id}_{method}'
                        drawpath,drawsha=save_draws(drawdir,fit.names,draws)
                        record['draw_path']=str(drawpath)
                        record['draw_sha256']=drawsha
                    intervals=dict(status='valid' if record['fit_success'] else 'exploratory_diagnostics_failed',intervals=json_safe(fit.intervals))
                record['runtime_seconds']=time.monotonic()-t
                commit_json(ROOT/'intervals'/f'appliances_{settings_id}_{method}.json',intervals)
                commit_json(path,record)
        except BudgetExpired:
            record.update(status='budget_interrupted',runtime_seconds=time.monotonic()-t)
            commit_json(ROOT/'diagnostics'/f'appliances_{method}_interrupted_{time.time_ns()}.json',record);break
        except Exception as exc:
            record.update(status='error',fit_success=False,error=str(exc),traceback=traceback.format_exc(),runtime_seconds=time.monotonic()-t)
            commit_json(path,record)
        outcomes.append(dict(method=method,status=record['status'],runtime_seconds=record['runtime_seconds']))
        print(outcomes[-1],flush=True)
    summary=dict(dataset='appliances',settings_id=settings_id,outcomes=outcomes,
        elapsed_seconds=time.monotonic()-started,budget_seconds=args.budget_seconds)
    commit_json(ROOT/'diagnostics'/f'appliances_execution_{time.time_ns()}.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
