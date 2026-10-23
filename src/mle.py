"""Profile and independent direct MLE fitting; no uncertainty or simulation.

See docs/DERIVATIONS.md (M1)-(M8). Boundary values are closure limits, not
positive-parameter MLEs. The Stage 3 likelihood remains the interior evaluator.
"""

from dataclasses import dataclass
from math import fsum, log
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, xlogy

from .censoring import BPCSExperiment
from .likelihood import log_censoring_constant, loglikelihood
from .model import Kumaraswamy


@dataclass
class MLEResult:
    """Numerical result; check BOTH converged and status before interpreting it.

    status: interior_mle, boundary_supremum, unbounded_likelihood,
    optimization_failed, or numerical_failure. Failed results are not MLEs.
    """

    a: float | None
    b_by_block: dict
    loglikelihood: float
    converged: bool
    status: str
    method: str
    message: str
    diagnostics: dict


def _positive(value, name):
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite real scalar")
    return float(value)


class _Problem:
    """Cached validated observations, shared by formulas, not an estimator."""

    def __init__(self, data):
        if not isinstance(data, BPCSExperiment):
            raise ValueError("data must be BPCSExperiment")
        self.data = data
        self.blocks = []
        for block in data.blocks:
            t = np.array([row.time for row in block.observations])
            self.blocks.append((block.scheme.block, t, -np.log(t),
                                np.array(block.scheme.withdrawals, dtype=float)+1,
                                np.array(list(block.cause_counts.values()), dtype=float)))
        self.M = sum(len(t) for _, t, _, _, _ in self.blocks)
        self.T = fsum(-float(u.sum()) for _, _, u, _, _ in self.blocks)
        self.logc = log_censoring_constant(data)
        self.boundary = tuple((label, c+1) for label, _, _, _, d in self.blocks
                              for c in range(2) if d[c] == 0)
        self.unbounded = all(len(t) == 1 for _, t, _, _, _ in self.blocks)

    def stats(self, a):
        a = _positive(a, "a")
        result = []
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for label, t, u, w, d in self.blocks:
                z = a*u
                logq = np.asarray(Kumaraswamy(a, 1).logsf(t))
                # v=-log q; its log remains usable when v itself underflows.
                logv = np.empty_like(z)
                tail = z > 36
                logv[tail] = -z[tail]  # relative correction < 1.2e-16
                logv[~tail] = np.log(-logq[~tail])
                logW = float(logsumexp(np.log(w)+logv))
                rho = np.exp(np.log(w)+logv-logW)
                # R=sum(w*p)/W, p=d(log q)/da. Group -z-logv to
                # avoid subtracting two large nearly equal log magnitudes.
                ratio = np.exp((-z-logv)+np.log(u)-logq)
                R = float(np.dot(rho, ratio))
                # delta=1-a*p. Series avoids cancellation near a*u=0.
                delta = np.empty_like(z)
                small = z < 1e-4
                delta[small] = z[small]/2-z[small]**2/12+z[small]**4/720
                delta[~small] = 1-z[~small]*np.exp(-z[~small])/(-np.expm1(-z[~small]))
                result.append((label, d, logW, float(logq.sum()), R, float(delta.sum())))
        return result

    def conditional(self, a):
        values = {}
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for label, d, logW, _, _, _ in self.stats(a):
                pair = tuple(float(np.exp(log(count)-logW)) if count else 0.0 for count in d)
                if any(count and value == 0 for count, value in zip(d, pair)):
                    raise FloatingPointError("conditional b is below float64 range")
                if not np.isfinite(sum(pair)):
                    raise FloatingPointError("conditional b sum exceeds float64 range")
                values[label] = pair
        return values

    def profile(self, a, include_constant=True):
        if not isinstance(include_constant, (bool, np.bool_)):
            raise ValueError("include_constant must be a boolean")
        terms = [self.logc if include_constant else 0, self.M*log(_positive(a, "a")),
                 (a-1)*self.T]
        for _, d, logW, H, _, _ in self.stats(a):
            terms.extend([float(xlogy(d, d).sum()), -float(d.sum())*logW,
                          -float(d.sum()), -H])
        value = fsum(terms)
        if not np.isfinite(value):
            raise FloatingPointError("profile log-likelihood is not representable")
        return value

    def profile_eta_score(self, a):
        terms = [a*self.T]
        for _, d, _, _, R, delta in self.stats(a):
            terms.extend([delta, float(d.sum())*a*R])
        value = fsum(terms)
        if not np.isfinite(value):
            raise FloatingPointError("profile score is not representable")
        return value

    def evaluate(self, a, b):
        """Stage 3 evaluation, with an EXACT linear correction for zero counts.

        Replace a closure zero by reference t=1/(1+W) temporarily. Since its
        count is zero, ell(0)=ell(t)+t*W. No epsilon estimate is substituted.
        """
        interior = dict(b)
        corrections = []
        for label, d, logW, _, _, _ in self.stats(a):
            pair = list(b[label])
            for c in range(2):
                if d[c] == 0:
                    if pair[c] != 0:
                        raise ValueError("inactive cause must be fixed to its closure value zero")
                    logref = -np.logaddexp(0, logW)
                    pair[c] = float(np.exp(logref))
                    corrections.append(float(np.exp(logref+logW)))
            interior[label] = tuple(pair)
        return fsum([loglikelihood(self.data, a, interior), *corrections])

    def log_gradient(self, a, b):
        terms = [a*self.T]
        bg = {}
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            for label, d, logW, _, R, delta in self.stats(a):
                pair = b[label]
                B = sum(pair)
                BW = float(np.exp(log(B)+logW))
                terms.extend([delta, BW*a*R])
                bg[label] = tuple(float(count-np.exp(log(value)+logW)) if value else 0.0
                                  for count, value in zip(d, pair))
        return fsum(terms), bg


def conditional_estimates(data, a):
    """Return d_ic/W_i(a), with exact closure zeros for absent causes."""
    return _Problem(data).conditional(a)


def profile_loglikelihood(data, a, *, include_constant=True):
    """Conditional maximum, or boundary supremum, over b at fixed a."""
    return _Problem(data).profile(a, include_constant)


def profile_score(data, a):
    """Derivative d ell_p / da (not with respect to log a)."""
    a = _positive(a, "a")
    return _Problem(data).profile_eta_score(a)/a


def score(data, a, b_by_block):
    """Raw interior scores; preserve Stage 3 validation and block-ID contract."""
    loglikelihood(data, a, b_by_block)  # validates the strictly positive parameters
    problem = _Problem(data)
    # Raw scores apply to all positive b, including an unobserved cause.
    ga, _ = problem.log_gradient(a, b_by_block)
    bscore = {}
    for label, d, logW, _, _, _ in problem.stats(a):
        W = float(np.exp(logW))
        bscore[label] = tuple(float(count/value-W)
                             for count, value in zip(d, b_by_block[label]))
    return {"a": ga/a, "b_by_block": bscore}


def _controls(maxiter, score_tol):
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    _positive(score_tol, "score_tol")


def _empty(problem, method, status, message, diagnostics=None):
    return MLEResult(None, {}, np.inf if status == "unbounded_likelihood" else np.nan,
                     False, status, method, message,
                     {"boundary_parameters": problem.boundary, **(diagnostics or {})})


def _finish(problem, method, a, b, success, message, diagnostics, score_tol):
    value = problem.evaluate(a, b)
    ga, gb = problem.log_gradient(a, b)
    norm = max(abs(ga), *(abs(v) for pair in gb.values() for v in pair))
    profiled = problem.profile(a)
    gap = profiled-value
    consistent = abs(gap) <= max(1e-8, score_tol*problem.M)
    converged = bool(success and norm <= score_tol*problem.M and consistent)
    status = ("boundary_supremum" if problem.boundary else "interior_mle") if converged else "optimization_failed"
    diagnostics.update({"log_parameter_score_norm": norm,
                        "score_threshold": score_tol*problem.M,
                        "boundary_parameters": problem.boundary,
                        "boundary_one_sided_scores": {
                            label: {c+1: -float(np.exp(logW)) for c in range(2) if d[c] == 0}
                            for label, d, logW, _, _, _ in problem.stats(a) if np.any(d == 0)},
                        "profile_loglikelihood_check": profiled,
                        "profile_minus_joint_loglikelihood": gap})
    return MLEResult(float(a), b, value, converged, status, method, message, diagnostics)


def fit_mle(data, *, initial_a=1.0, maxiter=200, max_bracket_steps=60, score_tol=1e-7):
    """Profile fit: adaptive sign bracket and Brent root in eta=log(a).

    Strict concavity in a yields a unique finite root when some block has m>=2.
    Numerical bracket exhaustion is reported as failure, not a boundary MLE.
    """
    problem = _Problem(data)
    initial_a = _positive(initial_a, "initial_a")
    _controls(maxiter, score_tol)
    if (isinstance(max_bracket_steps, bool) or not isinstance(max_bracket_steps, Integral)
            or max_bracket_steps < 1):
        raise ValueError("max_bracket_steps must be a positive integer")
    if problem.unbounded:
        return _empty(problem, "profile", "unbounded_likelihood",
                      "Every block has one failure: ell_p(a) tends to +infinity as a grows.")
    try:
        def objective(eta):
            return problem.profile_eta_score(float(np.exp(eta)))
        center = log(initial_a)
        low = high = center
        fcenter = objective(center)
        for steps in range(max_bracket_steps):
            if fcenter > 0:
                high += log(2)
                if objective(high) < 0:
                    break
            elif fcenter < 0:
                low -= log(2)
                if objective(low) > 0:
                    break
            else:
                break
        else:
            return _empty(problem, "profile", "optimization_failed", "No sign bracket within the configured expansion budget.",
                          {"log_a_bracket": (low, high)})
        if fcenter == 0:
            eta, iterations, calls, success = center, 0, 1, True
        else:
            eta, result = brentq(objective, low, high, xtol=1e-12, rtol=1e-12,
                                maxiter=maxiter, full_output=True, disp=False)
            iterations, calls, success = result.iterations, result.function_calls, result.converged
        a = float(np.exp(eta))
        b = problem.conditional(a)
        return _finish(problem, "profile", a, b, success,
                       "Profile root solved." if success else "Profile root iteration limit reached.",
                       {"iterations": iterations, "root_function_calls": calls,
                        "bracket_expansions": steps+1, "log_a_bracket": (low, high)}, score_tol)
    except (FloatingPointError, OverflowError) as exc:
        return _empty(problem, "profile", "numerical_failure", str(exc))


def fit_direct(data, *, starts=None, maxiter=1000, score_tol=1e-7):
    """Independent multistart joint optimization of active log parameters.

    No profile fit or conditional estimate supplies starting values or updates.
    starts: optional sequence of (a, b_by_block) strictly positive interior
    starts; absent causes are subsequently fixed to zero by the boundary model.
    """
    problem = _Problem(data)
    _controls(maxiter, score_tol)
    if problem.unbounded:
        return _empty(problem, "direct", "unbounded_likelihood",
                      "Every block has one failure; no finite maximizing parameter vector exists.")
    if starts is None:
        starts = [(a, {label: (1.0, 1.0) for label, *_ in problem.blocks})
                  for a in (.5, 1.0, 2.0, 5.0)]
    try:
        starts = tuple(starts)
    except TypeError as exc:
        raise ValueError("starts must be a sequence of (a,b_by_block) pairs") from exc
    if not starts:
        raise ValueError("starts must not be empty")
    active = [(label, c) for label, _, _, _, d in problem.blocks for c in range(2) if d[c] > 0]
    initial_vectors = []
    for start in starts:
        try:
            a, pairs = start
        except (TypeError, ValueError) as exc:
            raise ValueError("each start must be an (a,b_by_block) pair") from exc
        # Validate via the original interior likelihood before discarding any
        # supplied starting value for an unobserved cause.
        loglikelihood(data, a, pairs)
        initial_vectors.append(np.array([log(a), *(log(pairs[label][c]) for label, c in active)]))

    def unpack(vector):
        values = np.exp(vector)
        b = {label: [0.0, 0.0] for label, *_ in problem.blocks}
        for (label, c), value in zip(active, values[1:]):
            b[label][c] = float(value)
        return float(values[0]), {label: tuple(pair) for label, pair in b.items()}

    failures = 0
    def objective(vector):
        nonlocal failures
        try:
            a, b = unpack(vector)
            value = problem.evaluate(a, b)
            ga, gb = problem.log_gradient(a, b)
            gradient = np.array([ga, *(gb[label][c] for label, c in active)])
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("nonfinite joint objective")
            return -value, -gradient
        except (FloatingPointError, OverflowError, ValueError):
            failures += 1
            return np.inf, np.zeros(len(vector))

    runs = []
    best = None
    for vector in initial_vectors:
        result = minimize(objective, vector, jac=True, method="L-BFGS-B",
                          bounds=[(-700, 700)]*len(vector),
                          options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxls": 50})
        hit_guard = bool(np.any(np.abs(result.x) >= 700-1e-6))
        runs.append({"success": bool(result.success), "message": str(result.message),
                     "iterations": int(result.nit), "function_calls": int(result.nfev),
                     "loglikelihood": float(-result.fun), "hit_float_guard": hit_guard})
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    diagnostics = {"starts": runs, "numerical_rejections": failures, "log_parameter_guard": (-700, 700)}
    if best is None:
        return _empty(problem, "direct", "numerical_failure", "No finite joint candidate.", diagnostics)
    try:
        a, b = unpack(best.x)
        success = bool(best.success and not np.any(np.abs(best.x) >= 700-1e-6))
        return _finish(problem, "direct", a, b, success, str(best.message), diagnostics, score_tol)
    except (FloatingPointError, OverflowError) as exc:
        return _empty(problem, "direct", "numerical_failure", str(exc), diagnostics)
