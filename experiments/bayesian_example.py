"""Four Bayesian analyses on fixed teaching data, not a simulation study.

Run: python -m experiments.bayesian_example --output results/bayesian_teaching_example.json
The explicit priors below are illustrative, not calibrated expert elicitation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.stage2_example import load_example
from src.bayesian import GammaPrior, HierarchicalPrior, SamplingConfig, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = SamplingConfig(chains=args.chains, draws=args.draws, warmup=args.warmup, seed=args.seed)
    prior = HierarchicalPrior(a=GammaPrior(2, 2),
        kappa=(GammaPrior(3, 2), GammaPrior(3, 2)),
        lam=(GammaPrior(2, 2), GammaPrior(2, 2)))
    data = load_example()
    payload = {'description': 'Fixed seven-failure teaching data; not a performance or coverage study.',
               'configuration': asdict(config), 'hierarchical_prior': asdict(prior),
               'data': data.to_frame().to_dict(orient='records'), 'analyses': {}}
    for family, specification in [('independent', 'independent'), ('hierarchical', prior)]:
        for restricted in (False, True):
            name = family+('_within_block' if restricted else '_unrestricted')
            result = fit(data, prior=specification, restricted=restricted, config=config, times=(.5,))
            print(f'\n{name}: {result.status}', flush=True)
            record = {'status': result.status, 'converged': result.converged,
                      'estimates': result.estimates, 'moments': result.moments,
                      'intervals': {k: asdict(v) for k, v in result.intervals.items()},
                      'diagnostics': result.diagnostics, 'warnings': result.warnings,
                      'configuration': result.configuration}
            if result.draws.size:
                for parameter in result.names:
                    ci = result.intervals[parameter]
                    diag = result.diagnostics['parameters'][parameter]
                    print(f'  {parameter:10s} mean={result.estimates[parameter]} '
                          f'interval=[{ci.lower:.6g}, {ci.upper:.6g}] '
                          f'Rhat={diag["rhat"]:.4f} ESSbulk={diag["ess_bulk"]:.1f}', flush=True)
                record['reliability'] = {block: {name: {k: (asdict(v) if k == 'interval' else v)
                    for k, v in entry.items() if k != 'draws'} for name, entry in entries.items()}
                    for block, entries in result.reliability.items()}
            payload['analyses'][name] = record
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        print('\nSaved', args.output)
    print('Intervals are shortest contiguous sample intervals, not general disconnected HPD regions.')
    print('Diagnostic success is a check, not proof of convergence. Hyperpriors are illustrative.')


if __name__ == '__main__':
    main()
