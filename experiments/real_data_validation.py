"""Reproducible compatibility screening; never repairs or fits incompatible data.

Run with ``python -m experiments.real_data_validation`` after downloading the
source files listed below into results/real_data/original.
"""
import csv
import hashlib
import time
from collections import Counter
from pathlib import Path

from src.censoring import BlockData, BlockScheme, Observation
from .simulation_config import METHODS
from .simulation_storage import commit_json

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'real_data'
SOURCES = {
    'Hoel1972.txt': 'https://raw.githubusercontent.com/IowaBiostat/data-sets/main/Hoel1972/Hoel1972.txt',
    'Hoel1972_source.html': 'https://iowabiostat.github.io/data-sets/Hoel1972/Hoel1972.html',
    'melanoma.csv': 'https://vincentarelbundock.github.io/Rdatasets/csv/boot/melanoma.csv',
    'melanoma_source.html': 'https://search.r-project.org/CRAN/refmans/boot/html/melanoma.html',
}


def screen_hoel(rows, scale=1100.0):
    """Retain every source row; cancer vs other is an exhaustive regrouping.

    1100 days is an explicitly chosen analysis scale, NOT a verified biological
    upper bound. Scaled rows are only candidates and are never treated as valid
    exact continuous observations when ties remain.
    """
    allowed = {'Thymic lymphoma', 'Reticulum cell sarcoma', 'Other'}
    if len(rows) != 181 or {r['Group'] for r in rows} != {'Control', 'Germ-free'}:
        raise ValueError('Unexpected Hoel source dimensions/groups')
    processed = []
    for i, row in enumerate(rows, 1):
        if set(row) != {'Group', 'Cause', 'Time'} or row['Cause'] not in allowed:
            raise ValueError('Unexpected/missing source fields or cause')
        days = int(row['Time'])
        if not 0 < days < scale:
            raise ValueError('Source time outside proposed scale')
        processed.append(dict(source_row=i, block=row['Group'], original_days=days,
                              original_cause=row['Cause'], time=days/scale,
                              cause=2 if row['Cause'] == 'Other' else 1,
                              event_observed=True))
    blocks = []
    for group in ('Control', 'Germ-free'):
        subset = sorted((r for r in processed if r['block'] == group), key=lambda r:r['time'])
        counts = Counter(r['original_days'] for r in subset)
        ties = {str(t): n for t, n in sorted(counts.items()) if n > 1}
        # Source documents follow-up until death, so zero withdrawals is justified
        # for this attempted complete-data special case; no schedule is invented.
        try:
            BlockData(BlockScheme(group, len(subset), len(subset), (0,)*len(subset)),
                      tuple(Observation(group, r['time'], r['cause'], 0) for r in subset))
            compatible, error = True, None
        except ValueError as exc:
            compatible, error = False, str(exc)
        blocks.append(dict(block=group, observations=len(subset),
            original_cause_counts=dict(Counter(r['original_cause'] for r in subset)),
            grouped_cause_counts=dict(Counter(str(r['cause']) for r in subset)),
            observed_cause_1_proportion=sum(r['cause']==1 for r in subset)/len(subset),
            order_b1_le_b2_empirical_direction=sum(r['cause']==1 for r in subset)<=len(subset)/2,
            tied_days=ties, tied_excess_records=sum(n-1 for n in ties.values()),
            compatible=compatible, model_validation_error=error))
    return processed, dict(dataset='Hoel1972', observations=len(rows), blocks=blocks,
        compatible=all(b['compatible'] for b in blocks),
        censoring='All deaths observed according to source; no censoring indicator column.',
        cause_grouping={'1':['Thymic lymphoma','Reticulum cell sarcoma'], '2':['Other']},
        transformation={'candidate_scale_days':scale, 'formula':'time = original_days / 1100',
            'scientific_upper_bound_verified':False, 'purpose':'compatibility screening only'},
        fitted_observations=0, fitted_blocks=0,
        limitation='Tied day measurements violate current exact-time strictly ordered API. '
            'Requires verified finer times or a separately justified rounded/interval-time likelihood. '
            'No jitter, deletion, or regrouping of experimental blocks was applied. '
            'Scaling alone does not establish a Kumaraswamy model or independent latent risks.',
        model_fit_findings='Not assessed: incompatible data were not fitted.')


def main():
    started = time.monotonic()
    for directory in ('processed','estimates','intervals','diagnostics','validation'):
        (ROOT/directory).mkdir(parents=True, exist_ok=True)
    metadata = dict(schema_version=1, selected_candidate='Hoel1972',
        original_reference='Hoel DG (1972), Biometrics 28:475–488',
        transcription_reference='Kalbfleisch and Prentice (2002), via Iowa Biostatistics',
        files={name:dict(url=url, sha256=hashlib.sha256((ROOT/'original'/name).read_bytes()).hexdigest())
               for name,url in SOURCES.items()})
    commit_json(ROOT/'source_metadata.json', metadata)
    with (ROOT/'original'/'Hoel1972.txt').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    processed, audit = screen_hoel(rows)
    commit_json(ROOT/'processed'/'hoel_candidate.json', dict(status='screened_incompatible', rows=processed,
        transformation=audit['transformation'], cause_grouping=audit['cause_grouping']))
    commit_json(ROOT/'validation'/'hoel_compatibility.json', audit)
    with (ROOT/'original'/'melanoma.csv').open() as f:
        alternative = list(csv.DictReader(f))
    statuses = Counter(r['status'] for r in alternative)
    if len(alternative) != 205 or set(statuses) != {'1','2','3'}:
        raise ValueError('Unexpected alternative source structure')
    commit_json(ROOT/'validation'/'melanoma_compatibility.json', dict(dataset='boot::melanoma',
        observations=len(alternative), status_counts=dict(statuses), compatible=False,
        source_status_mapping={'1':'death from melanoma','2':'alive/right censored','3':'death from other causes'},
        reason='Arbitrary right censoring at individual follow-up times is not represented by the current completed BPCS likelihood. '
               'Censored patients were neither removed nor converted to withdrawals.',
        fitted_observations=0))
    for method in METHODS:
        record = dict(dataset='Hoel1972',method=method,status='not_run_incompatible_data',
            fit_success=False,reason=audit['limitation'],estimates=None,
            convergence=None,posterior_moments=None,intervals=None)
        commit_json(ROOT/'estimates'/f'hoel_{method}.json',record)
        commit_json(ROOT/'intervals'/f'hoel_{method}.json',dict(method=method,status='unavailable_incompatible_data',intervals=None))
    result = dict(status='blocked_no_verified_compatible_dataset', successful_fits=0,
        candidates_screened=['Hoel1972','boot::melanoma'],
        runtime_seconds=time.monotonic()-started, no_censoring_or_observations_fabricated=True)
    # Execution timing is a distinct append-only diagnostic, permitting reruns.
    commit_json(ROOT/'diagnostics'/f'compatibility_execution_{time.time_ns()}.json',result)
    print(result)


if __name__ == '__main__':
    main()
