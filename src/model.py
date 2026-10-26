"""Distribution primitives on (0, 1). See docs/DERIVATIONS.md (D1)-(D7).

Parameters are positive finite scalars; time arguments may be scalars or arrays.
CDF/survival extend to the real line. Density is defined as zero at and outside
the endpoints (endpoint values do not affect probabilities). Hazard is only
defined on the open support. NaN times are rejected rather than propagated.
"""

from dataclasses import dataclass

import numpy as np


def _positive(name, value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(value)


def _times(t):
    t = np.asarray(t, dtype=float)
    if np.any(np.isnan(t)):
        raise ValueError("times must not contain NaN")
    return t


def _return(value):
    return value.item() if value.ndim == 0 else value


def _log_one_minus_power(t, a, scale=1.0):
    """Stable scale*log(1-t**a) for interior times, including subnormal tails."""
    if scale == 0:
        return np.zeros_like(t)
    logt = np.log(t)
    with np.errstate(over="ignore", under="ignore"):
        z = a * logt
    out = np.empty_like(z)
    # If a*(-log t) is subnormal, forming it loses relative precision (or
    # rounds to zero). log(1-exp(-v)) = log(v) + O(v), where the correction
    # is far below log-scale precision here. Keep log(v) without multiplying.
    tiny = -z < np.finfo(float).tiny
    out[tiny] = np.log(a) + np.log(-logt[tiny])
    near = (z > -np.log(2.0)) & ~tiny
    out[near] = np.log(-np.expm1(z[near]))
    tail = ~(near | tiny)
    out[tail] = np.log1p(-np.exp(z[tail]))
    lost = tail & (-out < np.finfo(float).tiny)
    out *= scale
    # A power may underflow although its product with a large scale is
    # representable. In this tail log(1-exp(z))=-exp(z)*(1+O(exp(z))).
    with np.errstate(under="ignore"):
        out[lost] = -np.sign(scale)*np.exp(np.log(abs(scale))+z[lost])
    return out


@dataclass(frozen=True)
class Kumaraswamy:
    """Kw(a,b): latent lifetime or observed minimum with an appropriate b."""

    a: float
    b: float

    def __post_init__(self):
        object.__setattr__(self, "a", _positive("a", self.a))
        object.__setattr__(self, "b", _positive("b", self.b))

    def logsf(self, t):
        """Log P(T>t), used to retain small survival probabilities (D1)."""
        t = _times(t)
        out = np.full_like(t, -np.inf)
        out[t <= 0] = 0.0
        inside = (t > 0) & (t < 1)
        out[inside] = _log_one_minus_power(t[inside], self.a, self.b)
        return _return(out)

    def sf(self, t):
        """P(T>t): reliability at mission time t (D1)."""
        return _return(np.exp(self.logsf(t)))

    def cdf(self, t):
        """P(T<=t): cumulative probability of failure (D1)."""
        return _return(-np.expm1(self.logsf(t)))

    def logpdf(self, t):
        """Log lifetime density; no endpoint clipping or added epsilon (D2)."""
        t = _times(t)
        out = np.full_like(t, -np.inf)
        inside = (t > 0) & (t < 1)
        x = t[inside]
        out[inside] = (np.log(self.a) + np.log(self.b)
                       + (self.a - 1) * np.log(x)
                       + _log_one_minus_power(x, self.a, self.b - 1))
        return _return(out)

    def pdf(self, t):
        """Lifetime density, integrating to one over (0,1) (D2)."""
        return _return(np.exp(self.logpdf(t)))

    def hazard(self, t):
        """Instantaneous failure rate conditional on survival, f/S (D3)."""
        t = _times(t)
        if np.any((t <= 0) | (t >= 1)):
            raise ValueError("hazard requires 0 < t < 1")
        logh = (np.log(self.a) + np.log(self.b)
                + (self.a - 1) * np.log(t)
                - _log_one_minus_power(t, self.a))
        return _return(np.exp(logh))

    def median(self):
        """Median time to failure: solve S(t)=1/2 (D7)."""
        return float(np.exp(np.log(-np.expm1(-np.log(2.0) / self.b)) / self.a))


@dataclass(frozen=True)
class CompetingRisks:
    """One block: independent Kw(a,b1), Kw(a,b2) latent lifetimes.

    Construct one instance per facility using the shared a. This class does not
    impose a block count, a censoring design, or any order restriction.
    """

    a: float
    b1: float
    b2: float

    def __post_init__(self):
        for name in ("a", "b1", "b2"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if not np.isfinite(self.b1 + self.b2):
            raise ValueError("b1+b2 must be representable as a finite float")

    @property
    def minimum(self):
        """Distribution of X=min(T1,T2), Kw(a,b1+b2) (D4)."""
        return Kumaraswamy(self.a, self.b1 + self.b2)

    def _b(self, cause):
        if isinstance(cause, (bool, np.bool_)) or cause not in (1, 2):
            raise ValueError("cause must be 1 or 2")
        return self.b1 if cause == 1 else self.b2

    def cause_probability(self, cause):
        """P(delta=cause)=b_cause/(b1+b2), over the entire lifetime (D6)."""
        return self._b(cause) / (self.b1 + self.b2)

    def cause_logpdf(self, t, cause):
        """Log joint density of failure time and cause, log(f_k S_other) (D5).

        This is a subdensity, not the latent density or density given a cause.
        """
        b = self._b(cause)
        return _return(np.asarray(self.minimum.logpdf(t))
                       + np.log(b) - np.log(self.b1 + self.b2))

    def cause_pdf(self, t, cause):
        """g_k(t)dt=P(t<X<=t+dt, delta=k); integrates to P(delta=k)."""
        return _return(np.exp(self.cause_logpdf(t, cause)))

    def cause_cdf(self, t, cause):
        """Cumulative incidence P(X<=t, delta=k), distinct from latent CDF (D6)."""
        return _return(self.cause_probability(cause) * np.asarray(self.minimum.cdf(t)))
