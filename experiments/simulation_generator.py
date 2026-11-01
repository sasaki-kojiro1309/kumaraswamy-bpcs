"""Latent inverse sampling followed by actual survivor withdrawals.

No dataset is generated at import. Only observed data are serialized/passed on.
"""
import math
from pathlib import Path

import numpy as np

from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from .simulation_config import SCHEMA_VERSION, dataset_identity, digest
from .simulation_storage import CompatibilityError, commit_json, read_json

GENERATOR_VERSION = 'kw-bpcs-latent-v1'


class GenerationError(RuntimeError):
    pass


def inverse_kumaraswamy(uniforms, a, b):
    """Q(U) = exp(log(-expm1(log1p(-U)/b))/a), without clipping.

    Rounding to an endpoint or ties is a recorded generation failure, never
    repaired by perturbing times or by regenerating for an estimator.
    """
    if any(isinstance(v, bool) or not np.isscalar(v) or not np.isfinite(v) or v <= 0 for v in (a, b)):
        raise ValueError('positive finite scalar shapes are required')
    u = np.asarray(uniforms, dtype=float)
    if not np.all(np.isfinite(u)) or np.any((u <= 0) | (u >= 1)):
        raise ValueError('uniforms must be strictly inside (0,1)')
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            values = np.exp(np.log(-np.expm1(np.log1p(-u)/b))/a)
    except FloatingPointError as exc:
        raise GenerationError('inverse transform outside float64 range') from exc
    if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise GenerationError('inverse transform rounded to a support endpoint')
    return values


def _open_uniform(rng, size):
    values = rng.random(size)
    # NumPy uses [0,1). Reject exact endpoint uniforms only, not datasets.
    invalid = (values <= 0) | (values >= 1)
    for _ in range(100):
        if not np.any(invalid):
            return values
        values[invalid] = rng.random(int(invalid.sum()))
        invalid = (values <= 0) | (values >= 1)
    raise GenerationError('could not obtain open-interval uniforms')


def apply_progressive_censoring(scheme, failure_times, causes, rng):
    """Select next failure from CURRENT risk set, then remove random survivors.

    rng.choice sees unit indices only and is independent of both cause streams.
    No sorting-and-taking-first-m shortcut is used.
    """
    if not isinstance(scheme, BlockScheme):
        raise ValueError('scheme must be BlockScheme')
    times, labels = np.asarray(failure_times, dtype=float), np.asarray(causes)
    if (times.shape != (scheme.n,) or labels.shape != times.shape
            or not np.all(np.isfinite(times)) or np.any((times <= 0) | (times >= 1))
            or not np.all(np.isin(labels, (1, 2)))):
        raise ValueError('one interior failure time and cause 1/2 per original unit required')
    if len(np.unique(times)) != scheme.n:
        raise GenerationError('numerically tied unit minima; no silent tie-breaking')
    active = np.arange(scheme.n)
    rows = []
    for j, (r, expected_risk) in enumerate(zip(scheme.withdrawals, scheme.risk_sets)):
        if len(active) != expected_risk:
            raise GenerationError('risk-set accounting mismatch')
        unit = int(active[np.argmin(times[active])])
        rows.append(Observation(scheme.block, float(times[unit]), int(labels[unit]), r))
        active = active[active != unit]
        if r:
            removed = rng.choice(active, size=r, replace=False)
            active = active[~np.isin(active, removed)]
        if j+1 == scheme.m and len(active):
            raise GenerationError('terminal risk set was not exhausted')
    return BlockData(scheme, tuple(rows))


def generate_experiment(scenario, seed):
    """Generate independent latent causes; return ONLY validated observed blocks."""
    blocks = []
    for spec, block_seed in zip(scenario['blocks'], np.random.SeedSequence(seed).spawn(len(scenario['blocks']))):
        cause1, cause2, withdrawals = [np.random.default_rng(s) for s in block_seed.spawn(3)]
        scheme = BlockScheme(spec['block'], spec['n'], spec['m'], tuple(spec['withdrawals']))
        first = inverse_kumaraswamy(_open_uniform(cause1, scheme.n), scenario['a'], spec['b'][0])
        second = inverse_kumaraswamy(_open_uniform(cause2, scheme.n), scenario['a'], spec['b'][1])
        if np.any(first == second):
            raise GenerationError('numerically tied latent causes; no arbitrary cause assignment')
        labels = np.where(first < second, 1, 2)
        blocks.append(apply_progressive_censoring(scheme, np.minimum(first, second), labels, withdrawals))
    return BPCSExperiment(tuple(blocks))


def observed_payload(data):
    return [{'scheme': {'block': b.scheme.block, 'n': b.scheme.n, 'm': b.scheme.m,
                        'withdrawals': list(b.scheme.withdrawals)},
             'observations': [{'block': r.block, 'time': r.time, 'cause': r.cause,
                               'withdrawals': r.withdrawals} for r in b.observations]}
            for b in data.blocks]


def dataset_path(root, scenario_id, repetition):
    return Path(root)/'datasets'/scenario_id/f'rep_{repetition:06d}.json'


def make_dataset(scenario, repetition, master_seed, generator_implementation):
    uid, seed = dataset_identity(scenario, repetition, master_seed)
    data = generate_experiment(scenario, seed)
    return {'schema_version': SCHEMA_VERSION, 'generator_version': GENERATOR_VERSION,
            'dataset_id': uid, 'scenario_id': scenario['id'], 'repetition': repetition,
            'scenario_definition_hash': digest(scenario),
            'seed_information': {'master_seed': master_seed, 'dataset_seed': seed,
                'algorithm': 'SHA256-domain/SeedSequence-spawn/PCG64', 'numpy_version': np.__version__,
                'block_streams': 'block order; each spawns cause1, cause2, withdrawals'},
            'true_parameters': {'a': scenario['a'],
                'b_by_block': {b['block']: list(b['b']) for b in scenario['blocks']}},
            'number_of_blocks': len(data.blocks), 'blocks': observed_payload(data),
            'generator_implementation': generator_implementation}


def load_dataset(path, *, scenario=None, repetition=None, master_seed=None):
    payload, sha = read_json(path)
    allowed = {'schema_version', 'generator_version', 'dataset_id', 'scenario_id', 'repetition',
               'scenario_definition_hash', 'seed_information', 'true_parameters',
               'number_of_blocks', 'blocks', 'generator_implementation'}
    if set(payload) != allowed or payload['schema_version'] != SCHEMA_VERSION or payload['generator_version'] != GENERATOR_VERSION:
        raise CompatibilityError(f'unsupported observed dataset schema: {path}')
    blocks = []
    for b in payload['blocks']:
        if set(b) != {'scheme', 'observations'}:
            raise CompatibilityError('unexpected block fields, including latent data')
        if any(set(row) != {'block', 'time', 'cause', 'withdrawals'} for row in b['observations']):
            raise CompatibilityError('unexpected observation fields, including latent data')
        blocks.append(BlockData(BlockScheme(**b['scheme']), tuple(Observation(**r) for r in b['observations'])))
    data = BPCSExperiment(tuple(blocks))
    if len(data.blocks) != payload['number_of_blocks']:
        raise CompatibilityError('inconsistent block count')
    if scenario is not None:
        if payload['scenario_id'] != scenario['id'] or payload['scenario_definition_hash'] != digest(scenario):
            raise CompatibilityError('dataset does not match requested scenario')
        expected = {'a': scenario['a'], 'b_by_block': {b['block']: list(b['b']) for b in scenario['blocks']}}
        if payload['true_parameters'] != expected:
            raise CompatibilityError('true parameters do not match scenario')
        if len(scenario['blocks']) != len(data.blocks):
            raise CompatibilityError('scheme count mismatch')
        for spec, b in zip(scenario['blocks'], data.blocks):
            if b.scheme != BlockScheme(spec['block'], spec['n'], spec['m'], tuple(spec['withdrawals'])):
                raise CompatibilityError('saved scheme differs from scenario')
    if repetition is not None and payload['repetition'] != repetition:
        raise CompatibilityError('repetition mismatch')
    if scenario is not None and repetition is not None and master_seed is not None:
        uid, seed = dataset_identity(scenario, repetition, master_seed)
        if (payload['dataset_id'] != uid or payload['seed_information']['dataset_seed'] != seed
                or payload['seed_information']['master_seed'] != master_seed):
            raise CompatibilityError('dataset identity/seed mismatch')
    return data, payload, sha


def generate_and_save(root, scenario, repetition, master_seed, generator_implementation):
    """Commit before a caller may fit; saved observations are the source of truth."""
    path = dataset_path(root, scenario['id'], repetition)
    if path.exists():
        load_dataset(path, scenario=scenario, repetition=repetition, master_seed=master_seed)
        return path
    payload = make_dataset(scenario, repetition, master_seed, generator_implementation)
    commit_json(path, payload)
    # Read back the committed bytes through the same validation path as fitting.
    load_dataset(path, scenario=scenario, repetition=repetition, master_seed=master_seed)
    return path
