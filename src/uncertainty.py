"""Classical observed-information and marginal uncertainty; see derivations I1–I7.

Restricted probability intervals are exact binomial confidence sets intersected
with [0, 1/2]. Time-component and delta intervals are asymptotic approximations.
No ordinary constrained-b Wald inference is provided.
"""
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit
from scipy.stats import beta, norm

from .mle import MLEResult, _positive, _Problem
from .model import Kumaraswamy, CompetingRisks


class InformationError(ValueError):
    """Information cannot support a regular covariance calculation."""


@dataclass
class Interval:
    lower: float | None
    upper: float | None
    method: str
    confidence: float
    status: str = "available"
    note: str = ""


@dataclass
class InformationResult:
    names: tuple
    matrix: np.ndarray
    diagnostics: dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    estimates: dict
    standard_errors: dict
    intervals: dict
    time_names: tuple
    time_covariance: np.ndarray | None
    total_names: tuple
    covariance_total: np.ndarray | None
    original_names: tuple
    covariance_original: np.ndarray | None
    restricted: bool
    diagnostics: dict


def _confidence(value):
    value = _positive(value, "confidence")
    if value >= 1:
        raise ValueError("confidence must be less than 1")
    return value


def _parameters(data, a, b_by_block):
    problem = _Problem(data)
    a = _positive(a, "a")
    labels = tuple(b[0] for b in problem.blocks)
    if not isinstance(b_by_block, dict) or set(b_by_block) != set(labels):
        raise ValueError("parameter labels must exactly match data blocks")
    totals, probs = [], []
    for label in labels:
        if (not isinstance(b_by_block[label], (tuple, list, np.ndarray))
                or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                       for v in b_by_block[label])):
            raise ValueError("each block needs two real nonboolean b values")
        pair = np.asarray(b_by_block[label], dtype=float)
        if pair.shape != (2,) or not np.all(np.isfinite(pair)) or np.any(pair < 0):
            raise ValueError("each block needs two finite nonnegative b values")
        B = _positive(float(pair.sum()), "block total")
        totals.append(B)
        probs.append(float(pair[0]/B))
    return problem, a, labels, np.array(totals), np.array(probs)


def _scaled_derivatives(a, t):
    """Return a*L', a²*(-L''), 1-a*L', 1-a²*(-L'')."""
    z = a*(-np.log(t))
    if not np.all(np.isfinite(z)) or np.any(z <= 0):
        raise FloatingPointError("a*(-log t) outside numerical range")
    logq = np.asarray(Kumaraswamy(a, 1).logsf(t))
    ap = np.exp(np.log(z)-z-logq)
    aq = np.exp(2*np.log(z)-z-2*logq)
    dp, dq = 1-ap, 1-aq
    small = z < 1e-3
    v = z[small]
    dp[small] = v/2-v*v/12+v**4/720-v**6/30240
    dq[small] = v*v/12-v**4/240+v**6/6048
    return ap, aq, dp, dq


def observed_information(data, a, b_by_block, *, coordinates="total", time_only=False):
    """Analytical negative Hessian, not expected Fisher information.

    total order: (a, B[block]..., p[block]...). Original order:
    (a, b1[first], b2[first], ...). time_only permits exact zero cause values
    and returns (a, B...). Full information requires positive cause values.
    The matrix may be indefinite away from a maximum; inversion checks it.
    """
    problem, a, labels, Bs, ps = _parameters(data, a, b_by_block)
    if coordinates not in ("total", "original") or not isinstance(time_only, bool):
        raise ValueError("invalid coordinates or time_only")
    if time_only and coordinates != "total":
        raise ValueError("time_only requires total coordinates")
    if not time_only and np.any((ps == 0) | (ps == 1)):
        raise InformationError("full information is undefined at a zero-cause boundary")
    k = len(labels)
    J = np.zeros((1+k if time_only else 1+2*k,)*2)
    aa = 0.0
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        for i, ((label, t, _, w, d), B, p) in enumerate(zip(problem.blocks, Bs, ps)):
            ap, aq, _, dq = _scaled_derivatives(a, t)
            aa += float(np.sum(dq+B*w*aq))/a/a
            cross = -float(np.dot(w, ap))/a
            if coordinates == "total":
                j = 1+i
                J[0, j] = J[j, 0] = cross
                J[j, j] = d.sum()/B/B
                if not time_only:
                    J[1+k+i, 1+k+i] = d[0]/p/p+d[1]/(1-p)/(1-p)
            else:
                for c in range(2):
                    j = 1+2*i+c
                    b = b_by_block[label][c]
                    J[0, j] = J[j, 0] = cross
                    J[j, j] = d[c]/b/b
    J[0, 0] = aa
    if not np.all(np.isfinite(J)):
        raise FloatingPointError("observed information outside float64 range")
    names = (("a",)+tuple(f"B[{x}]" for x in labels)
             + (() if time_only else tuple(f"p[{x}]" for x in labels)))
    if coordinates == "original":
        names = ("a",)+tuple(f"b{c}[{x}]" for x in labels for c in (1, 2))
    return InformationResult(names, J, {"kind": "observed", "symmetric": True})


def covariance_from_information(information, *, max_condition=1e12):
    """Equilibrated Cholesky solve. Never regularizes or uses a pseudoinverse.

    Conditioning threshold applies after diagonal equilibration, to avoid
    rejecting a matrix merely because parameters use different units.
    """
    limit = _positive(max_condition, "max_condition")
    J = np.asarray(information.matrix if isinstance(information, InformationResult)
                   else information, dtype=float)
    if (J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] == 0
            or not np.all(np.isfinite(J))):
        raise InformationError("information must be a finite nonempty square matrix")
    if not np.allclose(J, J.T, rtol=1e-12, atol=0):
        raise InformationError("information must be symmetric")
    if np.any(np.diag(J) <= 0):
        raise InformationError("information must have positive diagonal")
    s = np.sqrt(np.diag(J))
    R = J/s[:, None]/s[None, :]
    condition = float(np.linalg.cond(R))
    if isinstance(information, InformationResult):
        information.diagnostics.update(raw_condition=float(np.linalg.cond(J)),
                                       equilibrated_condition=condition)
    try:
        factor = cho_factor(R, lower=True)
    except np.linalg.LinAlgError as exc:
        raise InformationError("information is not positive definite") from exc
    if not np.isfinite(condition) or condition > limit:
        raise InformationError("information is ill-conditioned after equilibration")
    V = cho_solve(factor, np.eye(len(s)))/s[:, None]/s[None, :]
    if not np.all(np.isfinite(V)) or np.any(np.diag(V) <= 0):
        raise InformationError("covariance outside numerical range")
    return (V+V.T)/2


def binomial_interval(d, m, *, confidence=.95, restricted=False):
    """Marginal Clopper–Pearson set; optional intersection with [0, 1/2]."""
    confidence = _confidence(confidence)
    if (any(not isinstance(x, Integral) or isinstance(x, (bool, np.bool_)) for x in (d, m))
            or m < 1 or not 0 <= d <= m or not isinstance(restricted, bool)):
        raise ValueError("require integer 0 <= d <= m, m >= 1 and boolean restricted")
    alpha = (1-confidence)/2
    lower = 0.0 if d == 0 else float(beta.ppf(alpha, d, m-d+1))
    upper = 1.0 if d == m else float(beta.isf(alpha, d+1, m-d))
    method = "Clopper-Pearson" + (" intersect [0, 0.5]" if restricted else "")
    if restricted and lower > .5:
        return Interval(None, None, method, confidence, "empty",
                        "Data conflict with the order at this confidence level; do not collapse to equality.")
    return Interval(lower, min(upper, .5) if restricted else upper, method, confidence,
                    note="Marginal coverage at least nominal under the assumed binomial model.")


def _wald(value, se, confidence, scale):
    if not np.isfinite(value) or not np.isfinite(se):
        return Interval(None, None, scale+" Wald", confidence, "unavailable",
                        "Estimate or standard error exceeds numerical range.")
    z = norm.isf((1-confidence)/2)
    if scale == "wald":
        bounds = (value-z*se, value+z*se)
    else:
        if scale == "log":
            center, error, inverse = np.log(value), se/value, np.exp
        else:
            if not 0 < value < 1:
                return Interval(None, None, "logit Wald", confidence, "unavailable",
                                "Estimate numerically saturates a support endpoint.")
            center, error, inverse = logit(value), se/value/(1-value), expit
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            bounds = inverse(np.array([center-z*error, center+z*error]))
        if not np.all(np.isfinite(bounds)) or (scale == "log" and bounds[0] == 0):
            return Interval(None, None, scale+" Wald", confidence, "unavailable",
                            "Interval exceeds floating-point range.")
    return Interval(float(bounds[0]), float(bounds[1]), scale+" Wald", confidence,
                    note="Asymptotic marginal approximation; small-sample coverage is not guaranteed.")


def _fit_parameters(data, fit, restricted):
    if (not isinstance(fit, MLEResult) or not fit.converged
            or fit.status not in ("interior_mle", "boundary_supremum")
            or not np.isfinite(fit.loglikelihood)):
        raise ValueError("inference requires a converged finite MLE or boundary supremum")
    known = fit.method in ("order_restricted_profile", "SLSQP_validation") or "restriction" in fit.diagnostics
    if restricted is None:
        restricted = known
    if not isinstance(restricted, bool) or (known and not restricted):
        raise ValueError("invalid restriction flag for fitted estimator")
    problem, a, labels, Bs, ps = _parameters(data, fit.a, fit.b_by_block)
    if problem.unbounded:
        raise ValueError("no finite identifiable shape estimate")
    for (_, d, logW, _, _, _), B, p in zip(problem.stats(a), Bs, ps):
        target = min(d[0]/d.sum(), .5) if restricted else d[0]/d.sum()
        if not np.isclose(p, target, atol=1e-7, rtol=1e-6) or abs(np.log(B)+logW-np.log(d.sum())) > 1e-5:
            raise ValueError("fit does not match data or declared restriction")
    if abs(problem.profile_eta_score(a)) > 1e-5*problem.M:
        raise ValueError("fit does not satisfy time-component score")
    return problem, a, labels, Bs, ps, restricted


def parameter_inference(data, fit, *, confidence=.95, restricted=None, max_condition=1e12):
    """SEs and marginal intervals; unsupported procedures are explicitly absent.

    Full covariance is supported only for unrestricted positive interior fits.
    Restricted and zero-cause fits retain regular time-component inference.
    Exact probability sets remain available when information inversion fails.
    """
    confidence = _confidence(confidence)
    _positive(max_condition, "max_condition")
    problem, a, labels, Bs, ps, restricted = _fit_parameters(data, fit, restricted)
    time = observed_information(data, a, fit.b_by_block, time_only=True)
    total_names = time.names+tuple(f"p[{x}]" for x in labels)
    original_names = ("a",)+tuple(f"b{c}[{x}]" for x in labels for c in (1, 2))
    estimates = dict(zip(time.names, [a, *Bs]))
    estimates.update(zip(total_names[len(time.names):], ps))
    estimates.update({f"b{c}[{label}]": fit.b_by_block[label][c-1]
                      for label in labels for c in (1, 2)})
    ses = dict.fromkeys(estimates)
    intervals = {name: {} for name in estimates}
    diagnostics = {"restricted": restricted, "probability_states": {},
                   "unsupported": [], "time_information": time.diagnostics}
    Vtime = Vq = Vo = None
    try:
        Vtime = covariance_from_information(time, max_condition=max_condition)
    except InformationError as exc:
        diagnostics["time_failure"] = str(exc)
    regular = not restricted and np.all((ps > 0) & (ps < 1))
    if regular:
        try:
            full = observed_information(data, a, fit.b_by_block)
            Vq = covariance_from_information(full, max_condition=max_condition)
            k = len(labels)
            G = np.zeros((1+2*k, 1+2*k))
            G[0, 0] = 1
            for i, (B, p) in enumerate(zip(Bs, ps)):
                G[1+2*i, 1+i], G[2+2*i, 1+i] = p, 1-p
                G[1+2*i, 1+k+i], G[2+2*i, 1+k+i] = B, -B
            Vo = G@Vq@G.T
            ses.update(zip(total_names, np.sqrt(np.diag(Vq))))
            ses.update(zip(original_names, np.sqrt(np.diag(Vo))))
            diagnostics["full_information"] = full.diagnostics
        except InformationError as exc:
            diagnostics["full_failure"] = str(exc)
    else:
        diagnostics["unsupported"].append("Full covariance and individual-b Wald inference for restricted or zero-cause fits")
    if Vtime is not None:
        ses.update(zip(time.names, np.sqrt(np.diag(Vtime))))
    for name, value in estimates.items():
        if ses[name] is not None:
            intervals[name]["wald"] = _wald(value, ses[name], confidence, "wald")
            scale = "logit" if name.startswith("p[") else "log"
            intervals[name][scale] = _wald(value, ses[name], confidence, scale)
    for (_, _, _, _, d), label, p in zip(problem.blocks, labels, ps):
        intervals[f"p[{label}]"]["exact"] = binomial_interval(int(d[0]), int(d.sum()),
                                                               confidence=confidence, restricted=restricted)
        diagnostics["probability_states"][label] = ("zero_boundary" if p in (0, 1) else
                                                     "active_equality" if restricted and p == .5 else "inactive")
    return InferenceResult(estimates, ses, intervals, time.names, Vtime, total_names,
                           Vq, original_names, Vo, restricted, diagnostics)


def reliability_gradients(a, B, p, t):
    """Analytical gradients in local (a,B,p) order, for interior time t."""
    a, B = _positive(a, "a"), _positive(B, "B")
    if not np.isscalar(p) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must lie in [0,1]")
    t = _positive(t, "t")
    if t >= 1:
        raise ValueError("reliability uncertainty requires 0 < t < 1")
    model = Kumaraswamy(a, B)
    S, h, median = model.sf(t), model.hazard(t), model.median()
    L = Kumaraswamy(a, 1).logsf(t)
    ap, _, dp, _ = _scaled_derivatives(a, np.array([t]))
    P = ap[0]/a
    gS = np.array([S*B*P, S*L, 0.])
    gh = np.array([h*(dp[0]/a+np.log(t)), h/B, 0.])
    z = np.log(2)/B
    ratio = np.exp(np.log(z)-z-np.log(-np.expm1(-z)))
    logmedian = np.log(-np.expm1(-z))/a
    gm = np.array([-median*logmedian/a, -median*ratio/a/B, 0.])
    F = model.cdf(t)
    gradients = {"survival": gS, "hazard": gh, "median": gm,
            "cause_1_probability": np.array([0., 0., 1.]),
            "cause_2_probability": np.array([0., 0., -1.]),
            "cause_1_incidence": -p*gS+np.array([0., 0., F]),
            "cause_2_incidence": -(1-p)*gS-np.array([0., 0., F])}
    if not all(np.all(np.isfinite(g)) for g in gradients.values()):
        raise FloatingPointError("reliability gradient outside float64 range")
    return gradients


def reliability_estimates(data, fit, t, *, restricted=None):
    """Fitted scalar-time reliability, reusing Stage 1 models, including closure limits."""
    _, a, labels, Bs, ps, _ = _fit_parameters(data, fit, restricted)
    t = _positive(t, "t")
    if t >= 1:
        raise ValueError("require 0 < t < 1")
    result = {}
    for label, B, p in zip(labels, Bs, ps):
        model = Kumaraswamy(a, B)
        if 0 < p < 1:
            competing = CompetingRisks(a, *fit.b_by_block[label])
            incidence = [competing.cause_cdf(t, c) for c in (1, 2)]
        else:
            incidence = [p*model.cdf(t), (1-p)*model.cdf(t)]
        result[label] = dict(survival=model.sf(t), hazard=model.hazard(t), median=model.median(),
                             cause_1_probability=float(p), cause_2_probability=float(1-p),
                             cause_1_incidence=incidence[0], cause_2_incidence=incidence[1])
    return result


def reliability_inference(data, fit, t, *, confidence=.95, restricted=None, max_condition=1e12):
    """Pointwise marginal delta intervals; probabilities use exact binomial sets.

    Return block -> quantity -> estimate, standard_error, interval, note.
    Restricted CIF uncertainty is unsupported; it involves nonregular p.
    These are pointwise intervals, not simultaneous reliability bands.
    """
    report = parameter_inference(data, fit, confidence=confidence, restricted=restricted,
                                 max_condition=max_condition)
    values = reliability_estimates(data, fit, t, restricted=report.restricted)
    _, a, labels, Bs, ps = _parameters(data, fit.a, fit.b_by_block)
    k = len(labels)
    result = {}
    for i, (label, B, p) in enumerate(zip(labels, Bs, ps)):
        gradients = reliability_gradients(a, B, p, t)
        entries = {}
        for name, value in values[label].items():
            entry = dict(estimate=float(value), standard_error=None, interval=None,
                         note="Unsupported: covariance unavailable or cause estimator nonregular.")
            is_probability = name.endswith("probability")
            if is_probability:
                ci = report.intervals[f"p[{label}]"]["exact"]
                if name.startswith("cause_2"):
                    ci = Interval(None if ci.upper is None else 1-ci.upper,
                                  None if ci.lower is None else 1-ci.lower,
                                  "complement of "+ci.method, ci.confidence, ci.status, ci.note)
                entry.update(interval=ci, note=ci.note)
            time_quantity = name in ("survival", "hazard", "median")
            V = report.time_covariance if time_quantity else report.covariance_total
            if V is not None:
                g = np.zeros(V.shape[0])
                local = gradients[name]
                g[0], g[1+i] = local[:2]
                if not time_quantity:
                    g[1+k+i] = local[2]
                variance = float(g@V@g)
                if not np.isfinite(variance) or variance < 0:
                    raise InformationError("invalid delta variance")
                se = float(np.sqrt(variance))
                entry.update(standard_error=se, note="Pointwise asymptotic delta approximation.")
                if not is_probability:
                    scale = "log" if name == "hazard" else "logit"
                    entry["interval"] = _wald(float(value), se, confidence, scale)
            entries[name] = entry
        result[label] = entries
    return result
