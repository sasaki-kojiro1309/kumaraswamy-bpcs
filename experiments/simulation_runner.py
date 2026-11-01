"""Explicit generate / fit / analyze / run entry points; NEVER runs on import.

This framework has been written without executing simulations or tests.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import traceback
import time

import numpy as np
import scipy

from .simulation_config import (ROOT, METHODS, SCHEMA_VERSION, ConfigurationError,
    digest, load_config, normalize_config, generation_definition, dataset_identity,
    fit_settings, seed_for, source_hashes, integer)
from .simulation_generator import (GENERATOR_VERSION, dataset_path, load_dataset,
                                   generate_and_save)
from .simulation_storage import (CompatibilityError, StorageError, TaskLocked,
    commit_json, read_json, task_lock, recover_locks, save_draws)
from .simulation_adapters import fit_method
from .simulation_budget import ExecutionBudget, BudgetExpired

GENERATION_SOURCES = ('experiments/simulation_config.py', 'experiments/simulation_storage.py',
    'experiments/simulation_generator.py', 'src/censoring.py', 'src/model.py')
FIT_SOURCES = ('experiments/simulation_config.py', 'experiments/simulation_storage.py',
    'experiments/simulation_budget.py',
    'experiments/simulation_adapters.py', 'experiments/simulation_runner.py',
    'src/censoring.py', 'src/model.py', 'src/likelihood.py', 'src/mle.py',
    'src/order_restricted.py', 'src/uncertainty.py', 'src/bayesian.py', 'src/bayesian_sampling.py')


def environment():
    return dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform())


def ensure_study(root, cfg, *, create=False):
    """Generation definition is invariant to priors, MCMC, methods, budget, path.

    Fitting on a different machine may use saved data; regenerating requires
    the identical generator implementation and NumPy version, or a new root.
    """
    root = Path(root)
    path = root/'run_manifest.json'
    definition = generation_definition(cfg)
    if not path.exists():
        if not create:
            raise CompatibilityError('No run manifest: generate datasets first, or select the correct output directory')
        snapshot_path = root/'config_snapshot.json'
        if snapshot_path.exists():
            old, _ = read_json(snapshot_path)
            if generation_definition(old['config']) != definition:
                raise CompatibilityError('incompatible initial configuration snapshot')
        else:
            commit_json(snapshot_path, {'schema_version': SCHEMA_VERSION, 'config': cfg})
        manifest = dict(schema_version=SCHEMA_VERSION, generation_definition=definition,
                        generation_definition_hash=digest(definition), generator_version=GENERATOR_VERSION,
                        generator_implementation=source_hashes(GENERATION_SOURCES),
                        generation_environment=environment())
        commit_json(path, manifest)
    manifest, _ = read_json(path)
    if (manifest['schema_version'] != SCHEMA_VERSION or manifest['generator_version'] != GENERATOR_VERSION
            or manifest['generation_definition_hash'] != digest(definition)):
        raise CompatibilityError('Generation configuration differs. Use a new output root; existing datasets are immutable.')
    if create and (manifest['generator_implementation'] != source_hashes(GENERATION_SOURCES)
                   or manifest['generation_environment']['numpy'] != np.__version__):
        raise CompatibilityError('Generator implementation/NumPy changed: use a new root to generate; fit-only can reuse saved data')
    return manifest


def generation_failure_path(root, sid, rep):
    return Path(root)/'diagnostics'/'generation_failures'/sid/f'rep_{rep:06d}.json'


def generate_stage(root, cfg, scenarios):
    manifest = ensure_study(root, cfg, create=True)
    counts = dict(saved=0, reused=0, generation_failed=0, locked=0)
    for scenario in scenarios:
        for rep in range(1, cfg['repetitions']+1):
            path = dataset_path(root, scenario['id'], rep)
            failure = generation_failure_path(root, scenario['id'], rep)
            if path.exists():
                load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
                counts['reused'] += 1
                continue
            if failure.exists():
                record, _ = read_json(failure)
                uid, _ = dataset_identity(scenario, rep, cfg['master_seed'])
                if record['dataset_id'] != uid:
                    raise CompatibilityError('generation failure belongs to a different configuration')
                counts['generation_failed'] += 1
                continue
            lock = Path(root)/'metadata'/'locks'/'generation'/f"{scenario['id']}_{rep:06d}.lock"
            try:
                with task_lock(lock):
                    if path.exists():
                        load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
                        counts['reused'] += 1
                        continue
                    try:
                        generate_and_save(root, scenario, rep, cfg['master_seed'], manifest['generator_implementation'])
                    except (CompatibilityError, StorageError):
                        raise
                    except Exception as exc:
                        uid, seed = dataset_identity(scenario, rep, cfg['master_seed'])
                        commit_json(failure, dict(schema_version=1, dataset_id=uid, scenario_id=scenario['id'],
                            repetition=rep, seed=seed, status='generation_failure', error_type=type(exc).__name__,
                            message=str(exc), traceback=traceback.format_exc()))
                        counts['generation_failed'] += 1
                    else:
                        counts['saved'] += 1
            except TaskLocked:
                counts['locked'] += 1
    return counts


def campaign(root, cfg, attempt=1):
    settings = dict(schema_version=1, settings=fit_settings(cfg),
                    implementation=source_hashes(FIT_SOURCES), environment=environment(), attempt=attempt)
    fit_id = digest(settings)[:24]
    commit_json(Path(root)/'metadata'/'fit_configs'/f'{fit_id}.json', dict(fit_id=fit_id, **settings))
    return fit_id


def result_path(root, fit_id, scenario_id, repetition, method):
    return Path(root)/'estimates'/'tasks'/fit_id/scenario_id/f'rep_{repetition:06d}_{method}.json'


def verify_result(path, fit_id, metadata, dataset_sha, method):
    result, sha = read_json(path)
    if (result.get('schema_version') != 1 or result.get('fit_id') != fit_id
            or result.get('dataset_id') != metadata['dataset_id']
            or result.get('dataset_sha256') != dataset_sha or result.get('method') != method
            or result.get('scenario_id') != metadata['scenario_id']
            or result.get('repetition') != metadata['repetition']):
        raise CompatibilityError(f'Result/dataset pairing mismatch: {path}')
    if result.get('draw_artifact'):
        import hashlib
        # Root is recovered from the task path: estimates/tasks/fit/scenario/file.
        root = Path(path).parents[4]
        draws = root/result['draw_artifact']['path']
        if hashlib.sha256(draws.read_bytes()).hexdigest() != result['draw_artifact']['sha256']:
            raise CompatibilityError(f'posterior draw artifact hash mismatch: {draws}')
    return result, sha


def fit_stage(root, cfg, scenarios, methods, *, attempt=1, budget=None, max_new_fits=None):
    budget = ExecutionBudget() if budget is None else budget
    ensure_study(root, cfg)
    # Preflight all requested files before any fitting. A recorded generation
    # failure is retained; an unexplained missing file is an actionable error.
    work = []
    for scenario in scenarios:
        for rep in range(1, cfg['repetitions']+1):
            path = dataset_path(root, scenario['id'], rep)
            if path.exists():
                load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
                work.append((scenario, rep, path))
            elif generation_failure_path(root, scenario['id'], rep).exists():
                failure, _ = read_json(generation_failure_path(root, scenario['id'], rep))
                if failure['dataset_id'] != dataset_identity(scenario, rep, cfg['master_seed'])[0]:
                    raise CompatibilityError('generation failure identity mismatch')
            else:
                raise CompatibilityError(f'Missing saved dataset: {path}. Run generate first; fit-only never generates.')
    fit_id = campaign(root, cfg, attempt)
    counts = dict(completed=0, reused=0, locked=0, budget_exhausted=False)
    for scenario, rep, path in work:
        # Load the exact same immutable observed-data object for all methods.
        data, metadata, sha = load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
        for method in methods:
            output = result_path(root, fit_id, scenario['id'], rep, method)
            if output.exists():
                verify_result(output, fit_id, metadata, sha, method)
                counts['reused'] += 1
                continue
            if budget.remaining() <= 0:
                counts['budget_exhausted'] = True
                return fit_id, counts
            if max_new_fits is not None and counts['completed'] >= max_new_fits:
                return fit_id, counts
            lock = Path(root)/'metadata'/'locks'/'fitting'/fit_id/f"{metadata['dataset_id']}_{method}.lock"
            try:
                with task_lock(lock):
                    if output.exists():
                        verify_result(output, fit_id, metadata, sha, method)
                        counts['reused'] += 1
                        continue
                    # Verify the committed disk bytes again immediately before
                    # use. No estimator receives truth, withdrawal identities,
                    # latent times, or unobserved eventual causes as data.
                    _, _, check_sha = load_dataset(path, scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
                    if check_sha != sha:
                        raise CompatibilityError('dataset changed after validation')
                    seed = seed_for(cfg['mcmc_master_seed'], 'kw-bpcs-mcmc-v1', metadata['dataset_id'], method)
                    try:
                        with budget.supervise():
                            bundle, draws = fit_method(data, scenario, cfg, method, seed)
                    except BudgetExpired:
                        commit_json(Path(root)/'diagnostics'/'budget_interruptions'/fit_id/
                            (f"{metadata['dataset_id']}_{method}_{time.time_ns()}.json"),
                            dict(dataset_id=metadata['dataset_id'],dataset_sha256=sha,
                                 scenario_id=scenario['id'],repetition=rep,method=method,
                                 status='budget_interrupted',fit_id=fit_id,mcmc_seed=seed))
                        counts['budget_exhausted'] = True
                        return fit_id, counts
                    bundle.update(fit_id=fit_id, dataset_id=metadata['dataset_id'], dataset_sha256=sha,
                                  dataset_path=str(path.relative_to(root)), scenario_id=scenario['id'],
                                  repetition=rep, ordering=scenario['ordering'], censoring=scenario['censoring'],
                                  scenario_definition_hash=digest(scenario), draw_artifact=None)
                    save = cfg['save_draws']
                    selected = (rep in save['repetitions']
                        or save['diagnostic_failures'] and bundle['status']=='diagnostics_failed'
                        or save['numerical_failures'] and bundle['status']=='numerical_failure')
                    if draws is not None and selected:
                        names, array = draws
                        target, draw_sha = save_draws(Path(root)/'draws'/fit_id/metadata['dataset_id']/method, names, array)
                        bundle['draw_artifact'] = {'path': str(target.relative_to(root)), 'sha256': draw_sha,
                                                   'names': list(names), 'shape': list(array.shape)}
                    commit_json(output, bundle)
                    counts['completed'] += 1
            except TaskLocked:
                counts['locked'] += 1
    return fit_id, counts


def choose_fit_id(root, requested=None):
    ids = sorted(p.stem for p in (Path(root)/'metadata'/'fit_configs').glob('*.json'))
    if requested is not None:
        if requested not in ids:
            raise CompatibilityError('unknown --fit-id; inspect metadata/fit_configs')
        return requested
    if len(ids) != 1:
        raise CompatibilityError('Specify --fit-id: there must be exactly one campaign for automatic analysis selection')
    return ids[0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'fit', 'analyze', 'run'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repetitions', type=int)
    parser.add_argument('--scenarios', nargs='+', help='explicit scenario IDs; no factorial expansion')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--prior-json', type=Path, help='replacement hierarchical shape-rate prior only')
    parser.add_argument('--mcmc-json', type=Path, help='JSON object of sampling-control overrides (no seed)')
    parser.add_argument('--chains', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--draws', type=int)
    parser.add_argument('--mcmc-master-seed', type=int)
    parser.add_argument('--attempt', type=int, default=1, help='new explicit fit campaign attempt; never overwrites old results')
    parser.add_argument('--fit-id', help='existing campaign for analyze mode only')
    parser.add_argument('--budget-seconds', type=float, help='elapsed fit budget; supervises in-flight fits on POSIX')
    parser.add_argument('--deadline-unix', type=float, help='absolute deadline shared across separate invocations')
    parser.add_argument('--max-new-fits', type=int, help='stop after this many newly completed method fits (benchmarking)')
    parser.add_argument('--recover-locks', action='store_true', help='only after ALL previous workers have stopped')
    return parser


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        budget = ExecutionBudget(args.budget_seconds, args.deadline_unix)
        if args.max_new_fits is not None:
            integer(args.max_new_fits, 'max_new_fits', 1)
        cfg = load_config(args.config)
        if args.repetitions is not None:
            cfg['repetitions'] = args.repetitions
        if args.prior_json:
            cfg['hierarchical_prior'] = json.loads(args.prior_json.read_text())
        if args.mcmc_json:
            cfg['mcmc'].update(json.loads(args.mcmc_json.read_text()))
        for field in ('chains', 'warmup', 'draws'):
            if getattr(args, field) is not None:
                cfg['mcmc'][field] = getattr(args, field)
        if args.mcmc_master_seed is not None:
            cfg['mcmc_master_seed'] = args.mcmc_master_seed
        cfg = normalize_config(cfg)
        integer(args.attempt, 'attempt', 1)
        if args.fit_id and args.mode != 'analyze':
            raise ConfigurationError('--fit-id is only for analyze mode')
        scenarios = [s for s in cfg['scenarios'] if args.scenarios is None or s['id'] in args.scenarios]
        if not scenarios or args.scenarios and set(args.scenarios)-{s['id'] for s in cfg['scenarios']}:
            raise ConfigurationError('unknown or empty scenario selection')
        methods = list(args.methods or cfg['methods'])
        if len(set(methods)) != len(methods):
            raise ConfigurationError('duplicate method selection')
        root = (args.output or Path(cfg['output_dir'])).resolve()
        ensure_study(root, cfg, create=args.mode in ('generate', 'run'))
        if args.recover_locks:
            recover_locks(root/'metadata'/'locks')
        request = {'mode': args.mode, 'config': cfg, 'scenario_ids': [s['id'] for s in scenarios],
                   'methods': methods, 'attempt': args.attempt, 'requested_fit_id': args.fit_id}
        commit_json(root/'metadata'/'requests'/(digest(request)+'.json'), request)
        if args.mode in ('generate', 'run'):
            report = generate_stage(root, cfg, scenarios)
            print('Generation status:', json.dumps(report, sort_keys=True))
            if report['locked'] and args.mode == 'run':
                raise TaskLocked('Generation tasks are still locked; resume after they complete')
        fit_id = None
        if args.mode in ('fit', 'run'):
            fit_id, report = fit_stage(root, cfg, scenarios, methods, attempt=args.attempt,
                                      budget=budget, max_new_fits=args.max_new_fits)
            print('Fit campaign:', fit_id, 'status:', json.dumps(report, sort_keys=True))
        if args.mode in ('analyze', 'run'):
            from .simulation_analysis import analyze_stage
            fit_id = fit_id or choose_fit_id(root, args.fit_id)
            report = analyze_stage(root, cfg, scenarios, methods, fit_id)
            print('Analysis:', report['analysis_id'], 'manifest:', report['manifest'])
        return 0
    except (ValueError, StorageError, OSError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
