"""Local, resumable completion of the original sixty-dataset submission.

Historical successes are selected first. Retries keep the original MCMC seed,
posterior and thresholds, increasing effort on a predetermined schedule.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time

from .simulation_config import load_config, METHODS, digest, seed_for
from .simulation_generator import load_dataset, dataset_path
from .simulation_runner import campaign, result_path, verify_result, environment
from .simulation_storage import commit_json, read_json, save_draws, task_lock, recover_locks
from .simulation_adapters import fit_method

ROOT = Path('results/simulation_submission')
OUT = Path('results/simulation_completion_v1')
OLD = 'c81a97c91afc340fd144ab3f'
EFFORT = ((1000, 2000), (2000, 8000), (4000, 32000))


def acceptable(b):
    """Independently enforce saved fit status, primary support and diagnostics."""
    if not b['fit_success'] or not b['converged'] or b['errors']:
        return False
    primary = [p for p in b['points'] if p['target_kind'] == 'primary']
    if len(primary) != 5 or any(not p['valid_point'] or p['estimate'] is None
                               or not 0 < p['estimate'] < float('inf') for p in primary):
        return False
    values = {p['parameter']: p['estimate'] for p in b['points']}
    if b['restricted'] and any(values[f'b1[{s}]'] > values[f'b2[{s}]'] for s in ('A', 'B')):
        return False
    if b['method'].startswith('bayes'):
        if b['status'] != 'diagnostics_passed' or b['posterior_propriety_status'] != 'proper':
            return False
        if any(p['first_moment'] != 'finite' or p['second_moment'] != 'finite' for p in primary):
            return False
        if not b['bayesian_diagnostics']:
            return False
        for d in b['bayesian_diagnostics']:
            if not (d.get('rhat') is not None and d['rhat'] <= 1.01
                    and d.get('ess_bulk', 0) >= 400 and d.get('ess_tail', 0) >= 400
                    and d.get('mcse_mean') is not None):
                return False
    return True


def candidates(cfg, scenario, rep, method):
    _, metadata, sha = load_dataset(dataset_path(ROOT, scenario['id'], rep),
        scenario=scenario, repetition=rep, master_seed=cfg['master_seed'])
    paths = [result_path(ROOT, OLD, scenario['id'], rep, method)]
    paths += sorted((OUT/'attempts'/scenario['id']).glob(f'rep_{rep:06d}_{method}_level_*.json'))
    found = []
    for path in paths:
        if path.exists():
            raw, _ = read_json(path)
            # Draw paths for new records remain relative to the original study.
            if path == paths[0]:
                b, _ = verify_result(path, OLD, metadata, sha, method)
            else:
                b = raw
                assert b['dataset_id'] == metadata['dataset_id'] and b['dataset_sha256'] == sha
                assert b['method'] == method and b['scenario_id'] == scenario['id'] and b['repetition'] == rep
            found.append((path, b))
    return metadata, sha, found


def inventory(cfg):
    rows = []
    for s in cfg['scenarios']:
        for rep in range(1, 21):
            for method in METHODS:
                meta, sha, found = candidates(cfg, s, rep, method)
                good = [(p, b) for p, b in found if acceptable(b)]
                selected = good[0] if good else found[-1] if found else None
                rows.append(dict(dataset_id=meta['dataset_id'], dataset_sha256=sha,
                    scenario_id=s['id'], repetition=rep, method=method,
                    state='complete' if good else 'invalid' if found else 'missing',
                    selected_path=str(selected[0]) if selected else None,
                    attempts=[str(p) for p, _ in found]))
    return rows


def complete_one(task):
    cfg, s, rep, method = task
    meta, sha, found = candidates(cfg, s, rep, method)
    if any(acceptable(b) for _, b in found):
        return (s['id'], rep, method, 'reused')
    # Known diagnostic failures skip directly to the longer, fixed-seed run.
    start_level = 1 if any(b['status'] == 'diagnostics_failed' for _, b in found) else 0
    for level in range(start_level, len(EFFORT)):
        output = OUT/'attempts'/s['id']/f'rep_{rep:06d}_{method}_level_{level}.json'
        if output.exists():
            b, _ = read_json(output)
            if acceptable(b):
                return (s['id'], rep, method, 'reused')
            continue
        local = deepcopy(cfg)
        local['mcmc'].update(warmup=EFFORT[level][0], draws=EFFORT[level][1])
        fit_id = campaign(ROOT, local)
        seed = seed_for(cfg['mcmc_master_seed'], 'kw-bpcs-mcmc-v1', meta['dataset_id'], method)
        with task_lock(output.with_suffix('.lock')):
            data, _, check_sha = load_dataset(dataset_path(ROOT, s['id'], rep),
                scenario=s, repetition=rep, master_seed=cfg['master_seed'])
            assert check_sha == sha
            b, draws = fit_method(data, s, local, method, seed)
            b.update(fit_id=fit_id, dataset_id=meta['dataset_id'], dataset_sha256=sha,
                scenario_id=s['id'], repetition=rep, ordering=s['ordering'], censoring=s['censoring'],
                scenario_definition_hash=digest(s), draw_artifact=None, effort_level=level)
            if draws is not None and (rep == 20 or not acceptable(b)):
                names, array = draws
                path, h = save_draws(ROOT/'draws'/'completion_v1'/meta['dataset_id']/f'{method}_level_{level}', names, array)
                b['draw_artifact'] = dict(path=str(path.relative_to(ROOT)), sha256=h,
                                          names=list(names), shape=list(array.shape))
            commit_json(output, b)
        print(json.dumps(dict(scenario=s['id'], rep=rep, method=method, level=level,
                              status=b['status'], seconds=b['runtime_seconds'])), flush=True)
        if acceptable(b):
            return (s['id'], rep, method, 'completed')
    return (s['id'], rep, method, 'unresolved')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['inventory', 'fit', 'analyze'])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--recover-locks', action='store_true', help='Only after interrupted workers have stopped; verifies local owners are dead')
    a = p.parse_args()
    cfg = load_config('configs/simulation_submission.json')
    OUT.mkdir(parents=True, exist_ok=True)
    if a.recover_locks:
        recover_locks(OUT/'attempts')
    if a.mode == 'analyze':
        from .completion_analysis import analyze
        analyze(cfg)
        from .completion_publication import compile_report
        compile_report()
        return
    rows = inventory(cfg)
    print(dict(Counter((r['method'], r['state']) for r in rows)), flush=True)
    if a.mode == 'inventory':
        path = OUT/'audits'/f'inventory_{time.time_ns()}.json'
        commit_json(path, dict(rows=rows, environment=environment(), cpu_count=os.cpu_count(),
            ram_bytes=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),
            effort_schedule=EFFORT, local_only=True))
        print(path)
    else:
        if not 1 <= a.workers <= 4:
            raise ValueError('Use 1 to 4 local worker processes')
        scenarios = {s['id']: s for s in cfg['scenarios']}
        tasks = [(cfg, scenarios[r['scenario_id']], r['repetition'], r['method'])
                 for r in rows if r['state'] != 'complete']
        # Expensive fits first to avoid a long serial tail.
        tasks.sort(key=lambda t: (not t[3].startswith('bayes_hierarchical'), t[2]))
        started = time.time()
        audit = dict(started_unix=started, workers=a.workers, task_count=len(tasks),
                     environment=environment(), config=cfg, effort_schedule=EFFORT,
                     execution='local ProcessPoolExecutor with spawn; no remote compute')
        commit_json(OUT/'audits'/f'fit_start_{int(started)}.json', audit)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(complete_one, t) for t in tasks]):
                print(future.result(), flush=True)
        commit_json(OUT/'audits'/f'fit_finish_{int(started)}.json',
                    dict(started_unix=started, finished_unix=time.time(), wall_seconds=time.time()-started))
        print('Final:', dict(Counter((r['method'], r['state']) for r in inventory(cfg))), flush=True)


if __name__ == '__main__':
    main()
