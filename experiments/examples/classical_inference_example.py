"""Deterministic two-block inference: python -m experiments.examples.classical_inference_example.

Seven hand-constructed failures teach the formulas, not asymptotic coverage.
"""
from experiments.examples.stage2_example import load_example
from src.mle import fit_mle
from src.order_restricted import fit_order_restricted_mle
from src.uncertainty import parameter_inference, reliability_inference


def main():
    data = load_example()
    for title, estimator in [('Unrestricted', fit_mle), ('Within-block restricted', fit_order_restricted_mle)]:
        fit = estimator(data)
        report = parameter_inference(data, fit)
        print(f'\n{title}: {fit.status}, log likelihood={fit.loglikelihood:.12f}')
        print('parameter       estimate         SE          95% interval')
        for name, value in report.estimates.items():
            se = report.standard_errors[name]
            ci = report.intervals[name].get('exact' if name.startswith('p[') else 'log')
            bounds = 'unsupported' if ci is None else f'[{ci.lower:.6f}, {ci.upper:.6f}] ({ci.method})' if ci.status == 'available' else ci.status
            print(f'{name:10s} {value:12.6f} {"unsupported" if se is None else f"{se:.6f}":>12s}  {bounds}')
        print('Probability states:', report.diagnostics['probability_states'])
        print('Time information diagnostics:', report.diagnostics['time_information'])
        print('Reliability at t=0.5 (intervals are marginal, pointwise):')
        for block, entries in reliability_inference(data, fit, .5).items():
            for name, entry in entries.items():
                ci = entry['interval']
                bounds = 'unsupported' if ci is None else ci.status if ci.status != 'available' else f'[{ci.lower:.6f}, {ci.upper:.6f}]'
                se = entry['standard_error']
                print(f'  {block} {name:20s}={entry["estimate"]:.6f}, SE={se}, CI={bounds}')
    print('\nTime estimates and uncertainty agree under both fits. Restricted individual-b and CIF uncertainty are unsupported.')
    print('Wald/delta intervals are approximations; this small teaching dataset does not establish coverage.')


if __name__ == '__main__':
    main()
