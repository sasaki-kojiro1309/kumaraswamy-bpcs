"""Explicit simulation specifications. Importing this module runs no experiment."""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

from src.censoring import BlockScheme
from src.bayesian import GammaPrior, HierarchicalPrior, SamplingConfig

SCHEMA_VERSION = 1
METHODS = (
    'mle_unrestricted', 'mle_within_block',
    'bayes_independent_unrestricted', 'bayes_independent_within_block',
    'bayes_hierarchical_unrestricted', 'bayes_hierarchical_within_block',
)
PAIRS = tuple((METHODS[i], METHODS[i+1]) for i in (0, 2, 4))
ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ConfigurationError(f'{name} must be an integer >= {minimum}')
    return int(value)


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f'{name} must be positive and finite')
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ConfigurationError(f'{name} must be a safe 1-80 character identifier')
    return value


def keys(value, allowed, required, name):
    if not isinstance(value, dict) or set(value)-set(allowed) or set(required)-set(value):
        raise ConfigurationError(f'{name}: required={sorted(required)}, allowed={sorted(allowed)}')


def withdrawal_schedule(n, m, plan):
    """Distributed: floor((j+1)w/m)-floor(jw/m), j=0,...,m-1.

    This cumulative-floor rule spreads remainders over the experiment, with
    prefix discrepancies below one withdrawal. Complete and m=1 designs work.
    """
    n, m = integer(n, 'n', 1), integer(m, 'm', 1)
    if m > n:
        raise ConfigurationError('m must not exceed n')
    w = n-m
    if plan == 'early':
        result = (w,)+(0,)*(m-1)
    elif plan == 'late':
        result = (0,)*(m-1)+(w,)
    elif plan == 'distributed':
        result = tuple(((j+1)*w)//m-(j*w)//m for j in range(m))
    else:
        raise ConfigurationError('plan must be early, late, or distributed')
    BlockScheme('check', n, m, result)
    return result


def hierarchical_prior(spec):
    keys(spec, ('a', 'kappa', 'lam'), ('a', 'kappa', 'lam'), 'hierarchical prior')
    def gamma(p):
        keys(p, ('shape', 'rate'), ('shape', 'rate'), 'Gamma prior')
        return GammaPrior(p['shape'], p['rate'])
    if any(not isinstance(spec[key], list) or len(spec[key]) != 2 for key in ('kappa', 'lam')):
        raise ConfigurationError('kappa and lam must each list two shape-rate priors')
    return HierarchicalPrior(gamma(spec['a']), tuple(map(gamma, spec['kappa'])),
                             tuple(map(gamma, spec['lam'])))


def normalize_config(raw):
    """Validate explicit scenarios without constructing a factorial study."""
    cfg = deepcopy(raw)
    allowed = ('schema_version', 'study_id', 'master_seed', 'mcmc_master_seed',
               'repetitions', 'output_dir', 'scenarios', 'methods', 'hierarchical_prior',
               'mcmc', 'classical', 'mission_times', 'include_boundary_suprema', 'save_draws')
    keys(cfg, allowed, allowed, 'simulation config')
    if cfg['schema_version'] != SCHEMA_VERSION or isinstance(cfg['schema_version'], bool):
        raise ConfigurationError('unsupported config schema_version')
    identifier(cfg['study_id'], 'study_id')
    for field in ('master_seed', 'mcmc_master_seed'):
        integer(cfg[field], field)
    integer(cfg['repetitions'], 'repetitions', 1)
    if not isinstance(cfg['output_dir'], str) or not cfg['output_dir'].strip():
        raise ConfigurationError('output_dir must be a nonempty path')
    if (not isinstance(cfg['methods'], list) or not cfg['methods']
            or any(m not in METHODS for m in cfg['methods'])
            or len(set(cfg['methods'])) != len(cfg['methods'])):
        raise ConfigurationError('methods must be a nonempty unique list of supported method IDs')
    if not isinstance(cfg['include_boundary_suprema'], bool):
        raise ConfigurationError('include_boundary_suprema must be boolean')
    if not isinstance(cfg['mission_times'], list):
        raise ConfigurationError('mission_times must be a list (possibly empty)')
    for t in cfg['mission_times']:
        if positive(t, 'mission time') >= 1:
            raise ConfigurationError('mission times must be strictly inside (0,1)')
    if len(set(cfg['mission_times'])) != len(cfg['mission_times']):
        raise ConfigurationError('mission_times must be distinct')
    classical_allowed = ('confidence', 'max_condition', 'initial_a', 'maxiter',
                         'max_bracket_steps', 'score_tol', 'constraint_tol')
    keys(cfg['classical'], classical_allowed, classical_allowed, 'classical controls')
    for field, value in cfg['classical'].items():
        if field in ('maxiter', 'max_bracket_steps'):
            integer(value, field, 1)
        else:
            positive(value, field)
    if cfg['classical']['confidence'] >= 1:
        raise ConfigurationError('classical confidence must be less than one')
    sampling_fields = ('chains', 'warmup', 'draws', 'slice_width', 'max_slice_steps',
                       'max_rhat', 'min_ess', 'probability')
    keys(cfg['mcmc'], sampling_fields, sampling_fields, 'MCMC controls (seed is managed separately)')
    SamplingConfig(**cfg['mcmc'], seed=0)  # Validation only, never sampling.
    hierarchical_prior(cfg['hierarchical_prior'])
    keys(cfg['save_draws'], ('repetitions', 'diagnostic_failures', 'numerical_failures'),
         ('repetitions', 'diagnostic_failures', 'numerical_failures'), 'draw storage')
    if not isinstance(cfg['save_draws']['repetitions'], list):
        raise ConfigurationError('draw repetitions must be a list')
    for rep in cfg['save_draws']['repetitions']:
        integer(rep, 'draw repetition', 1)
    for flag in ('diagnostic_failures', 'numerical_failures'):
        if not isinstance(cfg['save_draws'][flag], bool):
            raise ConfigurationError('draw flags must be boolean')
    if not isinstance(cfg['scenarios'], list) or not cfg['scenarios']:
        raise ConfigurationError('at least one explicit scenario is required')
    seen = set()
    for scenario in cfg['scenarios']:
        keys(scenario, ('id', 'ordering', 'censoring', 'a', 'blocks'),
             ('id', 'ordering', 'censoring', 'a', 'blocks'), 'scenario')
        sid = identifier(scenario['id'], 'scenario id')
        if sid in seen:
            raise ConfigurationError('duplicate scenario id')
        seen.add(sid)
        if not isinstance(scenario['ordering'], str) or not isinstance(scenario['censoring'], str):
            raise ConfigurationError('ordering/censoring are descriptive string labels')
        scenario['a'] = positive(scenario['a'], 'a')
        if not isinstance(scenario['blocks'], list) or not scenario['blocks']:
            raise ConfigurationError('scenario must contain at least one block')
        labels = set()
        for block in scenario['blocks']:
            keys(block, ('block', 'n', 'm', 'b', 'plan', 'withdrawals'),
                 ('block', 'n', 'm', 'b'), 'block')
            identifier(block['block'], 'block id')
            if block['block'] in labels:
                raise ConfigurationError('duplicate block id')
            labels.add(block['block'])
            if ('plan' in block) == ('withdrawals' in block):
                raise ConfigurationError('supply exactly one of plan or explicit withdrawals')
            if 'plan' in block:
                block['withdrawals'] = list(withdrawal_schedule(block['n'], block['m'], block.pop('plan')))
            scheme = BlockScheme(block['block'], block['n'], block['m'], tuple(block['withdrawals']))
            block['n'], block['m'] = scheme.n, scheme.m
            if not isinstance(block['b'], list) or len(block['b']) != 2:
                raise ConfigurationError('b must list two positive shapes')
            block['b'] = [positive(b, 'cause shape') for b in block['b']]
            if not math.isfinite(sum(block['b'])):
                raise ConfigurationError('cause total must be finite')
    return cfg


def load_config(path):
    return normalize_config(json.loads(Path(path).read_text()))


def generation_definition(cfg):
    """No prior, estimator, repetition budget, output path or MCMC setting."""
    return {'schema_version': SCHEMA_VERSION, 'master_seed': cfg['master_seed'],
            'scenarios': sorted(cfg['scenarios'], key=lambda s: s['id'])}


def seed_for(master, domain, *identity):
    # Full SHA-256 entropy is domain-separated; never use randomized hash().
    return int(digest({'master': master, 'domain': domain, 'identity': identity}), 16)


def dataset_identity(scenario, repetition, master_seed):
    integer(repetition, 'repetition', 1)
    seed = seed_for(master_seed, 'kw-bpcs-data-v1', scenario, repetition)
    uid = f"{scenario['id']}_rep_{repetition:06d}_{digest({'seed': seed})[:16]}"
    return uid, seed


def fit_settings(cfg):
    return {key: cfg[key] for key in ('mcmc_master_seed', 'hierarchical_prior', 'mcmc',
             'classical', 'mission_times', 'include_boundary_suprema', 'save_draws')}


def source_hashes(filenames):
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in filenames}
