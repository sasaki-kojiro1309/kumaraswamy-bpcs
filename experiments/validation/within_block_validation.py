"""Independent joint SLSQP verification, not a profile/pooling implementation.

Optimizes (log(a/a_scale), log B_i, p_i) with 0<=p_i<=1/2. The objective
is the row-expanded likelihood, including risk-set constants and zero-count
limits. No fitted starts, conditional estimates, or preset active sets are used.

Run: python -m experiments.validation.within_block_validation --output results/within_block_validation.json
"""

import argparse
from dataclasses import asdict
import json
from math import fsum, log
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

from experiments.examples.stage2_example import load_example
from src.censoring import BPCSExperiment
from src.mle import MLEResult
from src.model import Kumaraswamy


class _JointObjective:
    """Independent expanded objective and gradient for constrained verification."""

    def __init__(self, data):
        if not isinstance(data, BPCSExperiment):
            raise ValueError("data must be BPCSExperiment")
        self.blocks = []
        for block in data.blocks:
            t = np.array([r.time for r in block.observations])
            self.blocks.append((block.scheme.block, t, -np.log(t),
                                np.array(block.scheme.withdrawals)+1.,
                                np.array([block.cause_counts[1], block.cause_counts[2]])))
        self.M = sum(len(t) for _, t, *_ in self.blocks)
        self.T = fsum(-float(u.sum()) for _, _, u, _, _ in self.blocks)
        self.logc = fsum(log(y) for b in data.blocks for y in b.scheme.risk_sets)
        self.a_scale = self.M/(-self.T)  # time-only scale, not a fitted estimate

    def __call__(self, vector):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                a = float(np.exp(vector[0])*self.a_scale)
                values = [self.logc, self.M*log(a), (a-1)*self.T]
                ga_terms = [a*self.T]
                gradient = np.zeros(len(vector))
                for j, (_, t, u, w, d) in enumerate(self.blocks):
                    beta, p = vector[1+2*j:3+2*j]
                    B = float(np.exp(beta))
                    qlog = np.asarray(Kumaraswamy(a, 1).logsf(t))
                    z = a*u
                    ap = z*np.exp(-z)/(-np.expm1(-z))
                    delta = 1-ap
                    small = z < 1e-4
                    delta[small] = z[small]/2-z[small]**2/12+z[small]**4/720
                    values.extend([float(d.sum())*beta, float(xlogy(d[0], p)),
                                   float(xlogy(d[1], 1-p)), float(np.dot(w*B-1, qlog))])
                    ga_terms.extend([float(delta.sum()), B*float(np.dot(w, ap))])
                    gradient[1+2*j] = float(d.sum())+B*float(np.dot(w, qlog))
                    gradient[2+2*j] = ((d[0]/p if d[0] else 0.)
                                      -(d[1]/(1-p) if d[1] else 0.))
                gradient[0] = fsum(ga_terms)
                value = fsum(values)
                if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise FloatingPointError("nonfinite objective")
                return -value, -gradient
        except (FloatingPointError, OverflowError, ValueError):
            # Invalid trials cannot be certified: final objective/KKT checks
            # are independent of the SLSQP success flag.
            return np.inf, np.zeros(len(vector))


def fit_slsqp(data, *, maxiter=1000, score_tol=1e-6):
    """Focused independent validation fit with raw p bounds and joint updates.

    Numerical endpoint cleanup is limited to a zero-count p within 1e-10 of
    zero; objective and KKT checks are repeated at the exact closure endpoint.
    """
    problem = _JointObjective(data)
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if (not isinstance(score_tol, Real) or isinstance(score_tol, (bool, np.bool_))
            or not np.isfinite(score_tol) or score_tol <= 0):
        raise ValueError("score_tol must be a positive finite scalar")
    if all(len(t) == 1 for _, t, *_ in problem.blocks):
        return MLEResult(None, {}, np.inf, False, "unbounded_likelihood", "SLSQP_validation",
                         "All blocks have one failure: no finite maximum exists.", {})
    k = len(problem.blocks)
    offset = log(problem.a_scale)
    bounds = [(-700-offset, 700-offset)]+[b for _ in range(k) for b in ((-700, 700), (0, .5))]
    records, candidates = [], []
    for factor, B0, p0 in ((.1, .5, .2), (.7, 1., .35), (3., 2., .45)):
        start = np.array([log(factor), *([log(B0), p0]*k)])
        result = minimize(problem, start, jac=True, method="SLSQP", bounds=bounds,
                          options={"maxiter": maxiter, "ftol": 1e-12})
        v = result.x.copy()
        for j, (_, _, _, _, d) in enumerate(problem.blocks):
            if d[0] == 0 and 0 <= v[2+2*j] < 1e-10:
                v[2+2*j] = 0.
        loss, gradient = problem(v)
        raw = -gradient
        projected = raw.copy()
        for j in range(k):
            p = v[2+2*j]
            if p <= 1e-10:
                projected[2+2*j] = max(raw[2+2*j], 0.)
            elif p >= .5-1e-10:
                projected[2+2*j] = min(raw[2+2*j], 0.)
        norm = float(np.max(np.abs(projected)))
        feasible = bool(np.all(v[2::2] >= 0) and np.all(v[2::2] <= .5))
        hit_guard = bool(abs(v[0]+offset) >= 700-1e-6 or np.any(np.abs(v[1::2]) >= 700-1e-6))
        success = bool(result.success and np.isfinite(loss) and feasible and not hit_guard
                       and norm <= score_tol*problem.M)
        records.append({"solver_success": bool(result.success), "verified": success,
                        "message": str(result.message), "iterations": int(result.nit),
                        "function_calls": int(result.nfev),
                        "loglikelihood": float(-loss) if np.isfinite(loss) else None,
                        "projected_score_norm": norm, "feasible": feasible,
                        "hit_numerical_guard": hit_guard})
        if np.isfinite(loss):
            candidates.append((success, loss, v, norm, str(result.message)))
    if not candidates:
        return MLEResult(None, {}, np.nan, False, "numerical_failure", "SLSQP_validation",
                         "No finite joint candidate.", {"starts": records})
    verified = [c for c in candidates if c[0]]
    success, loss, v, norm, message = min(verified or candidates, key=lambda c: c[1])
    a = float(np.exp(v[0])*problem.a_scale)
    pairs, boundary = {}, []
    for j, (label, _, _, _, _) in enumerate(problem.blocks):
        B, p = float(np.exp(v[1+2*j])), float(v[2+2*j])
        pairs[label] = (B*p, B*(1-p))
        if p == 0:
            boundary.append((label, 1))
    status = ("boundary_supremum" if boundary else "interior_mle") if success else "optimization_failed"
    return MLEResult(a, pairs, float(-loss), success, status, "SLSQP_validation", message,
                     {"starts": records, "projected_score_norm": norm,
                      "score_threshold": score_tol*problem.M, "boundary_parameters": tuple(boundary),
                      "a_scale": problem.a_scale, "coordinates": "log(a/a_scale),log(B_i),p_i"})


def validation_report(data):
    """Reproducible teaching-data comparison; raise on a material disagreement."""
    from src.mle import fit_mle
    from src.order_restricted import fit_order_restricted_mle

    unrestricted = fit_mle(data)
    restricted = fit_order_restricted_mle(data)
    numerical = fit_slsqp(data)
    if not all(r.converged for r in (unrestricted, restricted, numerical)):
        raise RuntimeError("A validation fit failed; inspect the individual solver results")
    log_a_error = abs(log(numerical.a/restricted.a))
    pair_errors = [abs(x-y)/sum(restricted.b_by_block[label])
                   for label in restricted.b_by_block
                   for x, y in zip(numerical.b_by_block[label], restricted.b_by_block[label])]
    likelihood_error = abs(numerical.loglikelihood-restricted.loglikelihood)
    if log_a_error > 3e-5 or max(pair_errors) > 3e-5 or likelihood_error > 1e-8:
        raise RuntimeError("Independent SLSQP disagrees with the analytical estimator")
    return {"data_description": "Hand-constructed teaching data; not a simulation study",
            "restriction": "b_i1 <= b_i2 within each block only",
            "input": data.to_frame().to_dict("records"),
            "unrestricted": asdict(unrestricted), "restricted": asdict(restricted),
            "independent_slsqp": asdict(numerical),
            "comparison": {"absolute_log_a_difference": log_a_error,
                           "max_parameter_error_relative_to_block_total": max(pair_errors),
                           "absolute_loglikelihood_difference": likelihood_error}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validation_report(load_example())
    rendered = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Verified comparison saved to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
