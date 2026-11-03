"""Independent deterministic manuscript audit; no new statistical estimator.

Run: python -m experiments.validation.manuscript_audit --output results/manuscript_audit.json
Decimal likelihood differences use 100-digit arithmetic and h=1e-20.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal, localcontext
import hashlib
import json
from math import log
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import binom

from experiments.examples.stage2_example import load_example
from experiments.validation.within_block_validation import fit_slsqp
from src.likelihood import loglikelihood
from src.mle import fit_mle, fit_direct, profile_loglikelihood, profile_score
from src.order_restricted import fit_order_restricted_mle
from src.uncertainty import (observed_information, parameter_inference,
    reliability_gradients, reliability_inference, binomial_interval)


def decimal_loglikelihood(data, vector, coordinates):
    """Expanded ordered-row density, independent of production derivatives."""
    a, total = vector[0], Decimal(0)
    k = len(data.blocks)
    for i, block in enumerate(data.blocks):
        if coordinates == 'total':
            B, p = vector[1+i], vector[1+k+i]
            b1, b2 = B*p, B*(1-p)
        else:
            b1, b2 = vector[1+2*i:3+2*i]
            B = b1+b2
        for row, y in zip(block.observations, block.scheme.risk_sets):
            lt = Decimal.from_float(row.time).ln()
            lq = (1-(a*lt).exp()).ln()
            bc = b1 if row.cause == 1 else b2
            total += Decimal(y).ln()+a.ln()+bc.ln()+(a-1)*lt+((row.withdrawals+1)*B-1)*lq
    return total


def decimal_hessian(f, x):
    h = Decimal('1e-20')
    base = f(x)
    J = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(i, len(x)):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h; xm[i] -= h
                v = -(f(xp)-2*base+f(xm))/(h*h)
            else:
                values = []
                for si, sj in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    v = x.copy(); v[i] += si*h; v[j] += sj*h
                    values.append(f(v))
                v = -(values[0]-values[1]-values[2]+values[3])/(4*h*h)
            J[i,j] = J[j,i] = float(v)
    return J


def decimal_reliability(x, t):
    a, B, p = x
    lt = t.ln(); L = (1-(a*lt).exp()).ln()
    S = (B*L).exp()
    h = a*B*((a-1)*lt).exp()/(1-(a*lt).exp())
    med = ((1-(-Decimal(2).ln()/B).exp()).ln()/a).exp()
    return [S, h, med, p, 1-p, p*(1-S), (1-p)*(1-S)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_example()
    checks = {}
    with localcontext() as context:
        context.prec = 100
        errors = {}
        for coordinates in ('total', 'original'):
            for a in (.35, 1.3, 4.5):
                raw = [a,.9,1.4,.3,.65] if coordinates == 'total' else [a,.27,.63,.91,.49]
                x = [Decimal.from_float(v) for v in raw]
                if coordinates == 'total':
                    pairs = {s: (raw[1+i]*raw[3+i], raw[1+i]*(1-raw[3+i])) for i,s in enumerate(('A','B'))}
                else:
                    pairs = {'A': tuple(raw[1:3]), 'B': tuple(raw[3:5])}
                analytic = observed_information(data, a, pairs, coordinates=coordinates).matrix
                numeric = decimal_hessian(lambda v: decimal_loglikelihood(data,v,coordinates), x)
                err = float(np.max(np.abs(analytic-numeric)))
                assert np.allclose(analytic,numeric,rtol=5e-13,atol=2e-12), (coordinates,a,err)
                errors[f'{coordinates},a={a}'] = err
        checks['100_digit_hessian_max_absolute_errors'] = errors
        names = ('survival','hazard','median','cause_1_probability','cause_2_probability','cause_1_incidence','cause_2_incidence')
        errors = []
        for raw in ((1.3,.8,.3,.4),(.25,4.,.7,.9),(4.,.3,.5,.1)):
            x = [Decimal.from_float(v) for v in raw[:3]]
            t = Decimal.from_float(raw[3]); h = Decimal('1e-20')
            analytic = reliability_gradients(*raw[:3],raw[3])
            for j in range(3):
                xp,xm=x.copy(),x.copy(); xp[j]+=h; xm[j]-=h
                upper,lower=decimal_reliability(xp,t),decimal_reliability(xm,t)
                for name,up,down in zip(names,upper,lower):
                    derivative=float((up-down)/(2*h))
                    err=abs(analytic[name][j]-derivative)
                    assert np.isclose(analytic[name][j],derivative,rtol=5e-13,atol=1e-13)
                    errors.append(float(err))
        checks['100_digit_reliability_gradient_max_absolute_error'] = max(errors)
        # Independently differentiate the profiled expanded likelihood (not its score).
        errors=[]
        for af in (.02,.3,1.7,10.):
            a=Decimal.from_float(af); h=Decimal('1e-20')
            def profile(v):
                pairs=[]
                for b in data.blocks:
                    W=-sum(Decimal(r.withdrawals+1)*(1-(v*Decimal.from_float(r.time).ln()).exp()).ln() for r in b.observations)
                    pairs.extend([Decimal(b.cause_counts[c])/W for c in (1,2)])
                return decimal_loglikelihood(data,[v,*pairs],'original')
            derivative=float((profile(a+h)-profile(a-h))/(2*h))
            err=abs(profile_score(data,af)-derivative)
            assert np.isclose(profile_score(data,af),derivative,rtol=5e-12,atol=1e-12)
            assert np.isclose(profile_loglikelihood(data,af),float(profile(a)),atol=1e-11)
            errors.append(err)
        checks['100_digit_profile_score_max_absolute_error']=max(errors)
    U,R,D,N=fit_mle(data),fit_order_restricted_mle(data),fit_direct(data),fit_slsqp(data)
    assert all(f.converged for f in (U,R,D,N))
    checks['unrestricted_profile_direct_loglikelihood_difference']=abs(U.loglikelihood-D.loglikelihood)
    checks['restricted_profile_slsqp_loglikelihood_difference']=abs(R.loglikelihood-N.loglikelihood)
    checks['restricted_profile_slsqp_log_a_difference']=abs(log(R.a/N.a))
    pU,pR=parameter_inference(data,U),parameter_inference(data,R)
    Jo=observed_information(data,U.a,U.b_by_block,coordinates='original').matrix
    checks['covariance_transform_max_absolute_error']=float(np.max(np.abs(pU.covariance_original-np.linalg.solve(Jo,np.eye(5)))))
    assert checks['covariance_transform_max_absolute_error']<1e-12
    checks['time_covariance_invariance_max_absolute_error']=float(np.max(np.abs(pU.time_covariance-pR.time_covariance)))
    # Enumeration checks finitely many designs/probabilities; the proof is separate.
    coverages=[]
    for m in (1,3,10,40):
        sets=[binomial_interval(d,m,restricted=True) for d in range(m+1)]
        for p in np.linspace(0,.5,101):
            coverage=sum(binom.pmf(d,m,p) for d,ci in enumerate(sets) if ci.status!='empty' and ci.lower<=p<=ci.upper)
            coverages.append(float(coverage))
    checks['restricted_exact_minimum_enumerated_coverage']=min(coverages)
    assert min(coverages)>=.95-1e-12
    examples={}
    modules=['stage2_example','stage3_example','stage4_example','order_restricted_example','classical_inference_example','within_block_validation']
    root=Path(__file__).resolve().parents[2]
    for module in modules:
        command=[sys.executable,'-W','error','-m','experiments.'+module]
        run=subprocess.run(command,cwd=root,capture_output=True,text=True)
        assert run.returncode==0 and not run.stderr,(module,run.stderr)
        examples[module]={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
    sources=[p for folder in ('src','tests','experiments','docs','References') for p in sorted((root/folder).glob('*')) if p.is_file()]
    sources += [root/'Readme.md',root/'Report.tex',root/'requirements-lock.txt',root/'data/examples/two_blocks.json']
    output={'description':'Independent deterministic manuscript audit, 2026-11-01; no simulation or coverage-performance claim',
            'checks':checks,'fits':{name:asdict(fit) for name,fit in [('unrestricted',U),('restricted',R),('direct',D),('slsqp',N)]},
            'parameter_inference':{'unrestricted':asdict(pU),'restricted':asdict(pR)},
            'reliability_t_0_5':{'unrestricted':reliability_inference(data,U,.5),'restricted':reliability_inference(data,R,.5)},
            'examples':examples,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    def convert(x):
        if hasattr(x,'__dataclass_fields__'): return asdict(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,default=convert,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2))
    print('All six numerical demonstrations exited 0 with warnings treated as errors.')


if __name__=='__main__':
    main()
