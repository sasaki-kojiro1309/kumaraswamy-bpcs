"""Within-block b1 <= b2 MLE using the unchanged Stage 3 likelihood.

See docs/DERIVATIONS.md (R1)-(R6). Equality is allowed with positive b;
zero b1 is only a boundary supremum in the original parameter domain.
"""

from math import fsum, log

import numpy as np
from scipy.special import xlogy

from .likelihood import loglikelihood
from .mle import MLEResult, _positive, _Problem, fit_mle


def fit_order_restricted_mle(data, *, initial_a=1.0, maxiter=200,
                             max_bracket_steps=60, score_tol=1e-7,
                             constraint_tol=1e-10):
    """Fit b_i1 <= b_i2 independently in every validated experiment block.

    Reuses fit_mle's profile root: the restricted profile differs only by a
    data-dependent constant. Controls retain fit_mle's contracts. constraint_tol
    is a positive finite tolerance on (b1-b2)/(b1+b2), independent of scale.
    MLEResult.status='interior_mle' means positive-parameter attainment, even
    when an order equality is active; inspect diagnostics['active_constraints'].
    Failure to fit the unrestricted profile is propagated without claiming a
    restricted estimate. No uncertainty estimates or likelihood-ratio test.
    """
    constraint_tol = _positive(constraint_tol, "constraint_tol")
    base = fit_mle(data, initial_a=initial_a, maxiter=maxiter,
                   max_bracket_steps=max_bracket_steps, score_tol=score_tol)
    diagnostics = {"restriction": "b_i1 <= b_i2",
                   "constraint_tolerance": constraint_tol,
                   "unrestricted_status": base.status,
                   "unrestricted_diagnostics": base.diagnostics}
    if not base.converged:
        # In particular, don't inherit unrestricted zero-boundary labels:
        # cause-2 zeros can be removed by the order restriction.
        return MLEResult(None, {}, base.loglikelihood, False, base.status,
                         "order_restricted_profile", base.message, diagnostics)

    problem = _Problem(data)
    try:
        pairs, probabilities, active, forced, multipliers = {}, {}, {}, {}, {}
        boundary, losses, total_scores, residuals, p_scores = [], [], {}, {}, {}
        total_preservation = {}
        for label, d, logW, _, _, _ in problem.stats(base.a):
            m = float(d.sum())
            p = min(float(d[0]/m), .5)
            equality = bool(d[0] >= d[1])
            # Retain every already-feasible pair exactly, including equal
            # counts. Only strict violations need the pooling calculation.
            if d[0] > d[1]:
                with np.errstate(over="raise", invalid="raise", under="ignore"):
                    half = float(np.exp(log(m)-logW-log(2)))
                pair = (half, half)
            else:
                pair = base.b_by_block[label]
            B = sum(pair)
            if (not np.isfinite(B) or B <= 0
                    or any(v < 0 or (v == 0 and (c != 0 or d[0] != 0))
                           for c, v in enumerate(pair))):
                raise FloatingPointError("restricted parameters are outside float64 range")
            pairs[label] = pair
            probabilities[label] = p
            active[label] = equality
            forced[label] = bool(d[0] > d[1])
            multipliers[label] = {"upper_p": 2*float(d[0]-d[1]) if equality else 0.0,
                                  "lower_p": m if p == 0 else 0.0}
            derivative = -m if p == 0 else float(d[0]/p-d[1]/(1-p))
            p_scores[label] = (derivative-multipliers[label]["upper_p"]
                               +multipliers[label]["lower_p"])
            if p == 0:
                boundary.append((label, 1))
            residuals[label] = (pair[0]-pair[1])/B
            unrestricted_B = sum(base.b_by_block[label])
            total_preservation[label] = (B-unrestricted_B)/unrestricted_B
            total_scores[label] = m-float(np.exp(log(B)+logW))
            # The binomial log-likelihood penalty, with 0 log 0 = 0.
            losses.append(float(fsum((float(xlogy(d[0], d[0]/m)),
                                       float(xlogy(d[1], d[1]/m)),
                                       -float(xlogy(d[0], p)),
                                       -float(xlogy(d[1], 1-p))))))
        penalty = fsum(losses)
        expected = base.loglikelihood-penalty

        # Evaluate the SAME Stage 3 likelihood. At an actual b1=0 boundary,
        # replace it temporarily by b2 and add b2*W: its count is zero, so
        # the log likelihood is exactly linear in that coordinate. Positive
        # absent-cause parameters (all cause-1 blocks) must NOT be zeroed.
        interior = dict(pairs)
        corrections = []
        for label, d, logW, _, _, _ in problem.stats(base.a):
            if pairs[label][0] == 0:
                interior[label] = (pairs[label][1], pairs[label][1])
                corrections.append(float(np.exp(log(pairs[label][1])+logW)))
        value = fsum((loglikelihood(data, base.a, interior), *corrections))
        ga, _ = problem.log_gradient(base.a, pairs)
        norm = max(abs(ga), *(abs(v) for v in total_scores.values()),
                   *(abs(v) for v in p_scores.values()))
        threshold = score_tol*problem.M
        objective_tol = max(1e-8, threshold)
        valid = (all(v <= constraint_tol for v in residuals.values())
                 and all(abs(v) <= constraint_tol for v in total_preservation.values())
                 and norm <= threshold and abs(value-expected) <= objective_tol
                 and penalty >= -objective_tol
                 and value <= base.loglikelihood+objective_tol)
        diagnostics.update({"active_constraints": active,
                            "pooled_blocks": forced,
                            "cause_1_probabilities": probabilities,
                            "boundary_parameters": tuple(boundary),
                            "p_kkt_multipliers": multipliers,
                            "p_stationarity_residuals": p_scores,
                            "normalized_order_residuals": residuals,
                            "relative_total_preservation_residuals": total_preservation,
                            "log_total_scores": total_scores,
                            "log_a_score": ga,
                            "log_parameter_score_norm": norm,
                            "score_threshold": threshold,
                            "unrestricted_loglikelihood": base.loglikelihood,
                            "loglikelihood_penalty": penalty,
                            "restricted_profile_loglikelihood_check": expected,
                            "profile_minus_joint_loglikelihood": expected-value})
        status = ("boundary_supremum" if boundary else "interior_mle") if valid else "optimization_failed"
        return MLEResult(base.a, pairs, value, bool(valid), status,
                         "order_restricted_profile",
                         "Restricted profile verified." if valid else "Restricted numerical checks failed.",
                         diagnostics)
    except (FloatingPointError, OverflowError) as exc:
        return MLEResult(None, {}, np.nan, False, "numerical_failure",
                         "order_restricted_profile", str(exc), diagnostics)


def fit(data, *, restriction="b_i1 <= b_i2", **options):
    """Compatibility entry point for the previously reserved interface.

    Only the documented within-block restriction is supported.
    """
    if not isinstance(restriction, str) or restriction != "b_i1 <= b_i2":
        raise ValueError("only restriction='b_i1 <= b_i2' is supported")
    return fit_order_restricted_mle(data, **options)
