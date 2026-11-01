"""Six thin estimator adapters. No estimation occurs on import.

Only BPCSExperiment is passed to the established fitting functions. Truth is
used solely for evaluation records and never as a start/prior-fitting input.
"""
from dataclasses import asdict, is_dataclass
import math
from time import perf_counter
import traceback

import numpy as np

from src import bayesian
from src.mle import fit_mle
from src.order_restricted import fit_order_restricted_mle
from src.model import Kumaraswamy
from src.uncertainty import parameter_inference, reliability_inference, binomial_interval
from .simulation_config import METHODS, hierarchical_prior


def json_safe(value):
    """JSON diagnostics preserve missing/nonfinite values as null, never NaN."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def targets(scenario, times):
    values, kinds = {'a': scenario['a']}, {'a': 'primary'}
    for spec in scenario['blocks']:
        label, (b1, b2) = spec['block'], spec['b']
        B, p = b1+b2, b1/(b1+b2)
        for key, value, kind in [(f'b1[{label}]', b1, 'primary'), (f'b2[{label}]', b2, 'primary'),
                                  (f'B[{label}]', B, 'auxiliary'), (f'p[{label}]', p, 'auxiliary')]:
            values[key], kinds[key] = float(value), kind
        if times:
            model = Kumaraswamy(scenario['a'], B)
            entries = {'median': model.median(), 'cause_1_probability': p, 'cause_2_probability': 1-p}
            for t in times:
                suffix = repr(float(t))
                S, F = model.sf(t), model.cdf(t)
                entries.update({f'survival@{suffix}': S, f'hazard@{suffix}': model.hazard(t),
                                f'cause_1_incidence@{suffix}': p*F, f'cause_2_incidence@{suffix}': (1-p)*F})
            for key, value in entries.items():
                values[label+'/'+key], kinds[label+'/'+key] = float(value), 'reliability'
    return values, kinds


def _interval_record(parameter, truth, interval, family, kind, nominal, *, usable=True, reason='unsupported'):
    if interval is None:
        return dict(parameter=parameter, interval_family=family, interval_method=family,
                    interval_kind=kind, nominal_probability=nominal, lower=None, upper=None,
                    covered=None, width=None, available=False, valid_interval=False,
                    status=reason, source_status='unavailable', note='No supported interval returned.')
    source = asdict(interval) if is_dataclass(interval) else interval
    method = source.get('method', family)
    probability = source.get('confidence', source.get('probability', nominal))
    status = source.get('status', 'available')
    low, high = source.get('lower'), source.get('upper')
    empty = status == 'empty'
    finite = (low is not None and high is not None and math.isfinite(low)
              and math.isfinite(high) and low <= high)
    available = empty or (status == 'available' and finite)
    valid = bool(available and usable)
    return dict(parameter=parameter, interval_family=family, interval_method=method,
                interval_kind=kind, nominal_probability=probability,
                lower=None if empty else low, upper=None if empty else high,
                covered=0 if empty else int(low <= truth <= high) if finite else None,
                width=0.0 if empty else float(high-low) if finite else None,
                available=bool(available), valid_interval=valid,
                status=('exploratory_diagnostics_failed' if available and not usable else
                        'empty' if empty else 'available' if available else status if status != 'available' else 'invalid_endpoints'),
                source_status=status, note=source.get('note', ''),
                empirical_posterior_mass=source.get('empirical_mass'))


def _empty_bundle(method, scenario, cfg, seed):
    truth, kinds = targets(scenario, cfg['mission_times'])
    restricted = method.endswith('within_block')
    violation = any(b['b'][0] > b['b'][1] for b in scenario['blocks'])
    return dict(schema_version=1, method=method, restricted=restricted,
                restricted_model_misspecified=bool(restricted and violation),
                status='unexpected_error', source_status=None, converged=False,
                fit_success=False, attained_mle=False, boundary_supremum=False,
                posterior_propriety_status='not_applicable' if method.startswith('mle') else 'not_checked',
                runtime_seconds=None, mcmc_seed=seed if method.startswith('bayes') else None,
                mcmc=cfg['mcmc'] if method.startswith('bayes') else None,
                points=[dict(parameter=p, target_kind=kinds[p], truth=v, estimate=None,
                             valid_point=False, point_status='not_available',
                             first_moment='not_applicable' if method.startswith('mle') else 'unverified',
                             second_moment='not_applicable' if method.startswith('mle') else 'unverified')
                        for p, v in truth.items()], intervals=[], diagnostics={},
                bayesian_diagnostics=[], errors=[], active_constraints={}, pooled_blocks={})


def _failed_intervals(bundle, cfg):
    """Primary placeholders ensure failures/unsupported methods have denominators."""
    seen = {(r['parameter'], r['interval_family']) for r in bundle['intervals']}
    for point in bundle['points']:
        name = point['parameter']
        if bundle['method'].startswith('mle'):
            if name.startswith('p['):
                families = ('exact',) if bundle['restricted'] else ('exact', 'wald', 'logit')
            elif '/' in name:
                families = ('exact',) if name.endswith('probability') else ('log',) if 'hazard@' in name else ('logit',)
            else:
                families = ('wald', 'log')
            nominal = cfg['classical']['confidence']
            kind = 'confidence'
        else:
            families, nominal, kind = ('posterior_shortest',), cfg['mcmc']['probability'], 'credible'
        for family in families:
            if (name, family) in seen:
                continue
            bundle['intervals'].append(_interval_record(name, point['truth'], None, family, kind,
                nominal, reason='unsupported' if bundle['converged'] else bundle['status']))


def _classical(data, scenario, cfg, method, bundle):
    restricted = bundle['restricted']
    options = {k: cfg['classical'][k] for k in ('initial_a', 'maxiter', 'max_bracket_steps', 'score_tol')}
    if restricted:
        fit = fit_order_restricted_mle(data, constraint_tol=cfg['classical']['constraint_tol'], **options)
    else:
        fit = fit_mle(data, **options)
    bundle.update(source_status=fit.status, status=fit.status, converged=fit.converged,
                  fit_success=bool(fit.converged and fit.status == 'interior_mle'),
                  attained_mle=bool(fit.converged and fit.status == 'interior_mle'),
                  boundary_supremum=fit.status == 'boundary_supremum',
                  diagnostics=json_safe(fit.diagnostics),
                  active_constraints=fit.diagnostics.get('active_constraints', {}),
                  pooled_blocks=fit.diagnostics.get('pooled_blocks', {}))
    # Exact cause sets need only observed counts; no information inversion/MLE.
    nominal = cfg['classical']['confidence']
    truth = {p['parameter']: p['truth'] for p in bundle['points']}
    for block in data.blocks:
        name = f'p[{block.scheme.block}]'
        interval = binomial_interval(block.cause_counts[1], block.scheme.m,
                                     confidence=nominal, restricted=restricted)
        bundle['intervals'].append(_interval_record(name, truth[name], interval, 'exact', 'confidence', nominal))
    if not fit.converged or fit.status not in ('interior_mle', 'boundary_supremum'):
        return None
    values = {'a': fit.a}
    for label, pair in fit.b_by_block.items():
        values.update({f'b1[{label}]': pair[0], f'b2[{label}]': pair[1],
                       f'B[{label}]': sum(pair), f'p[{label}]': pair[0]/sum(pair)})
    valid = fit.status == 'interior_mle' or cfg['include_boundary_suprema']
    for row in bundle['points']:
        if row['parameter'] in values:
            row.update(estimate=values[row['parameter']], valid_point=valid,
                       point_status='valid' if valid else 'boundary_supremum_excluded')
    try:
        report = parameter_inference(data, fit, confidence=nominal, restricted=restricted,
                                     max_condition=cfg['classical']['max_condition'])
        bundle['diagnostics']['inference'] = json_safe(report.diagnostics)
        for name, intervals in report.intervals.items():
            for family, interval in intervals.items():
                if family == 'exact':  # Already collected even if inversion fails.
                    continue
                bundle['intervals'].append(_interval_record(name, truth[name], interval, family, 'confidence', nominal))
    except Exception as exc:
        bundle['errors'].append(dict(stage='parameter_inference', error_type=type(exc).__name__,
                                     message=str(exc), traceback=traceback.format_exc()))
    for index, t in enumerate(cfg['mission_times']):
        try:
            reliability = reliability_inference(data, fit, t, confidence=nominal,
                                                restricted=restricted, max_condition=cfg['classical']['max_condition'])
            for label, entries in reliability.items():
                for quantity, entry in entries.items():
                    timed = quantity in ('survival', 'hazard', 'cause_1_incidence', 'cause_2_incidence')
                    if not timed and index:
                        continue
                    name = label+'/'+quantity+('@'+repr(float(t)) if timed else '')
                    for row in bundle['points']:
                        if row['parameter'] == name:
                            row.update(estimate=entry['estimate'], valid_point=valid,
                                       point_status='valid' if valid else 'boundary_supremum_excluded')
                    family = ('exact' if quantity.endswith('probability') else
                              'log' if quantity == 'hazard' else 'logit')
                    bundle['intervals'].append(_interval_record(name, truth[name], entry['interval'], family,
                                                               'confidence', nominal))
        except Exception as exc:
            bundle['errors'].append(dict(stage='reliability_inference', time=t,
                error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    return None


def _bayes(data, scenario, cfg, method, bundle, seed):
    independent = method.startswith('bayes_independent')
    prior = 'independent' if independent else hierarchical_prior(cfg['hierarchical_prior'])
    check = bayesian.posterior_conditions(data, prior=prior, restricted=bundle['restricted'])
    bundle['posterior_propriety_status'] = check['status']
    bundle['diagnostics']['propriety'] = json_safe(check)
    for point in bundle['points']:
        moment = check['moments'].get(point['parameter'], {})
        point['first_moment'] = moment.get('first', 'unverified')
        point['second_moment'] = moment.get('second', 'unverified')
    if check['proper'] is not True:
        bundle['status'] = 'improper_posterior' if check['proper'] is False else 'unsupported_prior_data'
        bundle['source_status'] = check['status']
        for point in bundle['points']:
            point['first_moment'] = check['moments'].get(point['parameter'], {}).get('first', 'unverified')
            point['second_moment'] = check['moments'].get(point['parameter'], {}).get('second', 'unverified')
            point['point_status'] = bundle['status']
        return None
    config = bayesian.SamplingConfig(**cfg['mcmc'], seed=seed)
    fit = bayesian.fit(data, prior=prior, restricted=bundle['restricted'], config=config,
                       times=tuple(cfg['mission_times']))
    bundle.update(source_status=fit.status, status=fit.status, converged=fit.converged,
                  fit_success=bool(fit.converged and fit.status == 'diagnostics_passed'))
    bundle['diagnostics'].update(json_safe(fit.diagnostics))
    bundle['diagnostics']['configuration'] = json_safe(fit.configuration)
    bundle['diagnostics']['warnings'] = list(fit.warnings)
    estimates, intervals, moments = dict(fit.estimates), dict(fit.intervals), dict(fit.moments)
    diagnostic_entries = dict(fit.diagnostics.get('parameters', {}))
    for label, entries in fit.reliability.items():
        for quantity, entry in entries.items():
            name = label+'/'+quantity
            estimates[name], intervals[name], moments[name] = entry['estimate'], entry['interval'], entry['moments']
            diagnostic_entries[name] = entry['diagnostics']
        # p_i is an actually returned draw-wise posterior quantity, even if no
        # mission times were requested. Do not invent a B_i posterior interval.
        if 'cause_1_probability' in entries:
            entry = entries['cause_1_probability']
            name = f'p[{label}]'
            estimates[name], intervals[name], moments[name] = entry['estimate'], entry['interval'], entry['moments']
    for spec in scenario['blocks']:
        label = spec['block']; names = (f'b1[{label}]', f'b2[{label}]')
        if all(estimates.get(n) is not None and moments.get(n, {}).get('first') == 'finite' for n in names):
            estimates[f'B[{label}]'] = sum(estimates[n] for n in names)
            moments[f'B[{label}]'] = {'first': 'finite', 'second': 'finite' if all(moments[n]['second']=='finite' for n in names) else 'unverified'}
    for row in bundle['points']:
        name, truth = row['parameter'], row['truth']
        moment = moments.get(name, {})
        first, second = moment.get('first', 'unverified'), moment.get('second', 'unverified')
        value = estimates.get(name)
        valid = bool(bundle['fit_success'] and first == 'finite' and value is not None and math.isfinite(value))
        row.update(estimate=float(value) if value is not None and math.isfinite(value) else None,
                   valid_point=valid, first_moment=first, second_moment=second,
                   point_status='valid' if valid else 'mcmc_diagnostic_failure' if fit.status=='diagnostics_failed' else
                                'posterior_mean_'+first if first != 'finite' else fit.status)
        # A proper posterior can have a valid interval without a finite mean.
        bundle['intervals'].append(_interval_record(name, truth, intervals.get(name),
            'posterior_shortest', 'credible', cfg['mcmc']['probability'],
            usable=bundle['fit_success'], reason='unsupported' if bundle['fit_success'] else fit.status))
    for name, diag in diagnostic_entries.items():
        moment = moments.get(name, {})
        bundle['bayesian_diagnostics'].append(json_safe(dict(parameter=name, **diag,
            first_moment=moment.get('first', 'unverified'), second_moment=moment.get('second', 'unverified'))))
    bundle['posterior_propriety_status'] = 'proper'
    return (fit.names, fit.draws) if fit.draws.size else None


def fit_method(data, scenario, cfg, method, seed):
    """Capture per-method failures; return JSON record and optional in-memory draws.

    Caller MUST have persisted and reloaded the dataset before calling this.
    A failed estimator never asks the generator for another sample.
    """
    if method not in METHODS:
        raise ValueError('unsupported method')
    bundle = _empty_bundle(method, scenario, cfg, seed)
    start, draws = perf_counter(), None
    try:
        draws = (_classical(data, scenario, cfg, method, bundle) if method.startswith('mle')
                 else _bayes(data, scenario, cfg, method, bundle, seed))
    except (FloatingPointError, OverflowError, bayesian.SamplingError) as exc:
        bundle['status'] = 'numerical_failure'
        bundle['errors'].append(dict(stage='fit', error_type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    except bayesian.PosteriorError as exc:
        bundle['status'] = 'unsupported_prior_data'
        bundle['errors'].append(dict(stage='fit', error_type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    except Exception as exc:
        bundle['status'] = 'unexpected_error'
        bundle['errors'].append(dict(stage='fit', error_type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    if bundle['status'] in ('numerical_failure', 'unsupported_prior_data', 'unexpected_error'):
        bundle['fit_success'] = False
        bundle['converged'] = False
        for row in bundle['points']:
            row['valid_point'] = False
        for row in bundle['intervals']:
            if row['interval_kind'] == 'credible':
                row['valid_interval'] = False
    bundle['runtime_seconds'] = perf_counter()-start
    _failed_intervals(bundle, cfg)
    if not bundle['bayesian_diagnostics'] and method.startswith('bayes'):
        bundle['bayesian_diagnostics'].append(dict(parameter='__fit__', first_moment='unverified', second_moment='unverified'))
    return json_safe(bundle), draws
