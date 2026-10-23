"""Kumaraswamy BPCS likelihood, conditional on fixed completed-block schemes.

See docs/DERIVATIONS.md (L1)-(L8). Defaults include C=product of risk sets,
giving the density of ordered failure times and their observed cause sequence.
include_constant=False omits ONLY C, retaining all observed-time factors.
No prior, parameter order restriction, optimizer, or missing-cause model is used.
"""

from collections.abc import Mapping
from math import fsum, isfinite, log
from numbers import Real

import numpy as np

from .censoring import BPCSExperiment, BlockData
from .model import CompetingRisks


def log_censoring_constant(data):
    """Log product of pre-failure risk sets for BlockData or BPCSExperiment."""
    if isinstance(data, BlockData):
        blocks = (data,)
    elif isinstance(data, BPCSExperiment):
        blocks = data.blocks
    else:
        raise ValueError("data must be BlockData or BPCSExperiment")
    return fsum(log(y) for block in blocks for y in block.scheme.risk_sets)


def _model(a, b1, b2):
    # Keep parameter validation in the Stage 1 class, with a strict real-scalar
    # input contract here for future numerical optimization/sampling adapters.
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
           for v in (a, b1, b2)):
        raise ValueError("parameters must be positive finite real scalars")
    return CompetingRisks(a, b1, b2)


def _finite_sum(terms):
    try:
        result = fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise FloatingPointError("log-likelihood exceeds float64 arithmetic range") from exc
    if not isfinite(result):
        raise FloatingPointError("log-likelihood is not finite in float64 arithmetic")
    return result


def block_loglikelihood(block, a, b1, b2, *, include_constant=True):
    """Evaluate one block at fixed positive parameters; return a float.

    Reuses Stage 1 joint cause log densities and minimum log survival, and Stage 2
    validated data. Invalid inputs raise ValueError. Unrepresentable logarithmic
    arithmetic raises FloatingPointError, distinct from invalid parameters.
    """
    if not isinstance(block, BlockData):
        raise ValueError("block must be BlockData")
    if not isinstance(include_constant, (bool, np.bool_)):
        raise ValueError("include_constant must be a boolean")
    model = _model(a, b1, b2)
    terms = []
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for row in block.observations:
            terms.append(model.cause_logpdf(row.time, row.cause))
            if row.withdrawals:  # avoid evaluating a mathematically absent term
                terms.append(row.withdrawals * model.minimum.logsf(row.time))
    if include_constant:
        terms.append(log_censoring_constant(block))
    return _finite_sum(terms)


def _exp(log_value):
    # An ordinary likelihood can legitimately underflow to zero or overflow to
    # infinity; retain the unclipped value and use log-likelihood for inference.
    with np.errstate(over="ignore", under="ignore"):
        return float(np.exp(log_value))


def block_likelihood(block, a, b1, b2, *, include_constant=True):
    """Ordinary block likelihood; use block_loglikelihood for computation."""
    return _exp(block_loglikelihood(block, a, b1, b2,
                                    include_constant=include_constant))


def loglikelihood(data, a, b_by_block, *, include_constant=True):
    """Joint log-likelihood with ONE shared a and explicit block-ID mapping.

    b_by_block maps every data block ID to a (b1,b2) pair; missing or extra
    keys are rejected. Block order and mapping insertion order need not match.
    No flattened parameter-vector convention is imposed on later estimators.
    """
    if not isinstance(data, BPCSExperiment):
        raise ValueError("data must be BPCSExperiment")
    if not isinstance(b_by_block, Mapping):
        raise ValueError("b_by_block must map block IDs to (b1,b2) pairs")
    if set(b_by_block) != {block.scheme.block for block in data.blocks}:
        raise ValueError("parameter block IDs must exactly match data block IDs")
    terms = []
    for block in data.blocks:
        pair = b_by_block[block.scheme.block]
        if isinstance(pair, (str, bytes, Mapping)):
            raise ValueError("each block must have a (b1,b2) pair")
        try:
            b1, b2 = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("each block must have a (b1,b2) pair") from exc
        terms.append(block_loglikelihood(block, a, b1, b2,
                                         include_constant=include_constant))
    return _finite_sum(terms)


def likelihood(data, a, b_by_block, *, include_constant=True):
    """Ordinary joint likelihood, including censoring constants by default."""
    return _exp(loglikelihood(data, a, b_by_block,
                             include_constant=include_constant))
