"""Focused one-dimensional kernels and multi-chain diagnostics for Bayesian BPCS.

No lifetime simulator. See docs/BAYESIAN_DERIVATIONS.md for targets/limitations.
"""
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import gamma, norm, rankdata


class SamplingError(RuntimeError):
    """A computation cannot be certified within configured numerical budgets."""


def slice_step(log_density, current, rng, *, width=1., max_steps=1000):
    """Neal stepping-out/shrinkage update on a real coordinate.

    Expands until BOTH ends are outside the slice. Budget exhaustion raises;
    it never silently substitutes an unverified or fixed-width proposal.
    """
    if not np.isfinite(width) or width <= 0 or max_steps < 1:
        raise ValueError('positive width and step budget required')
    value = float(log_density(current))
    if not np.isfinite(value):
        raise SamplingError('slice initial state has nonfinite density')
    height = value-rng.exponential()
    left = current-width*rng.random(); right = left+width
    for _ in range(max_steps):
        if log_density(left) <= height:
            break
        left -= width
    else:
        raise SamplingError('slice left stepping-out budget exhausted')
    for _ in range(max_steps):
        if log_density(right) <= height:
            break
        right += width
    else:
        raise SamplingError('slice right stepping-out budget exhausted')
    for _ in range(max_steps):
        proposal = rng.uniform(left, right)
        if log_density(proposal) >= height:
            return float(proposal)
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise SamplingError('slice shrinkage budget exhausted')


def _log_gamma_tail(shape, x, upper=False):
    """Log regularized Gamma tail, with series/continued fraction past underflow."""
    if x == 0:
        return 0. if upper else -np.inf
    if np.isinf(x):
        return -np.inf if upper else 0.
    value = gamma.logsf(x, shape) if upper else gamma.logcdf(x, shape)
    if np.isfinite(value):
        return float(value)
    logprefactor = shape*np.log(x)-x-gammaln(shape)
    if not upper:
        term = total = 1./shape
        for j in range(1, 10001):
            term *= x/(shape+j); total += term
            if abs(term) < abs(total)*2e-15:
                return float(logprefactor+np.log(total))
    else:
        # Modified Lentz fraction for Q; this branch is only the far upper tail.
        tiny = 1e-300
        b = x+1-shape; c = 1/tiny; d = 1/b; h = d
        for j in range(1, 10001):
            an = -j*(j-shape); b += 2
            d = an*d+b; c = b+an/c
            if abs(d) < tiny: d = tiny
            if abs(c) < tiny: c = tiny
            d = 1/d; change = d*c; h *= change
            if abs(change-1) < 2e-15:
                return float(logprefactor+np.log(h))
    raise SamplingError('Gamma log-tail expansion failed')


def _logdiff(a, b):
    if b == -np.inf:
        return a
    if not a > b:
        raise SamplingError('truncation probability unresolved at float64 precision')
    return float(a+np.log(-np.expm1(b-a)))


def truncated_gamma(shape, rate, rng, *, lower=0., upper=np.inf):
    """Inverse-tail draw, shape-rate Gamma restricted to (lower,upper).

    Chooses lower/upper tails to avoid cancellation; far-tail inversion uses
    logarithmic tail probabilities, not rejection or epsilon bounds.
    """
    if (not np.isfinite(shape) or not np.isfinite(rate) or shape <= 0 or rate <= 0
            or not np.isfinite(lower) or lower < 0 or np.isnan(upper) or upper <= lower):
        raise ValueError('invalid Gamma parameters or truncation bounds')
    lo, hi = lower*rate, upper*rate
    if not np.isfinite(lo) or (np.isfinite(upper) and not np.isfinite(hi)):
        raise SamplingError('scaled Gamma truncation outside numerical range')
    use_upper = gamma.cdf(hi, shape) > .5
    if use_upper:
        highprob = _log_gamma_tail(shape, lo, True)
        lowprob = _log_gamma_tail(shape, hi, True)
    else:
        highprob = _log_gamma_tail(shape, hi)
        lowprob = _log_gamma_tail(shape, lo)
    logwidth = _logdiff(highprob, lowprob)
    logtarget = np.logaddexp(lowprob, logwidth-rng.exponential())
    # Fast path when the probability itself is representable.
    if logtarget > -700:
        prob = np.exp(logtarget)
        x = gamma.isf(prob, shape) if use_upper else gamma.ppf(prob, shape)
    else:
        x = np.nan
    if not np.isfinite(x) or not lo < x < hi:
        # In log x, support/range failures are explicit, not artificial truncation.
        left = np.log(lo) if lo else -744.
        right = np.log(hi) if np.isfinite(hi) else 709.
        def root(z):
            return _log_gamma_tail(shape, np.exp(z), use_upper)-logtarget
        if not left < right or np.signbit(root(left)) == np.signbit(root(right)):
            raise SamplingError('Gamma quantile outside representable range')
        x = float(np.exp(brentq(root, left, right, xtol=2e-13, rtol=1e-14)))
    draw = float(x/rate)
    if not np.isfinite(draw) or not lower < draw < upper:
        raise SamplingError('Gamma draw not representable inside truncation interval')
    return draw


@dataclass(frozen=True)
class ShortestInterval:
    lower: float
    upper: float
    probability: float
    empirical_mass: float
    method: str = 'shortest contiguous sample interval (unimodal HPD approximation)'


def shortest_interval(draws, probability=.95):
    """Minimum-width closed interval covering at least ceil(probability*N) draws.

    A multimodal HPD region need not be contiguous. Endpoints and method are
    in the supplied parameterization; HPD is not transformation invariant.
    """
    x = np.sort(np.asarray(draws, dtype=float).ravel())
    if len(x) < 2 or not np.all(np.isfinite(x)) or not 0 < probability < 1:
        raise ValueError('finite sample of length >=2 and 0<probability<1 required')
    count = max(1, ceil(probability*len(x)))
    widths = x[count-1:]-x[:len(x)-count+1]
    j = int(np.argmin(widths)); low, high = x[j], x[j+count-1]
    mass = float(np.mean((x >= low) & (x <= high)))
    return ShortestInterval(float(low), float(high), probability, mass)


def _split(x):
    half = x.shape[1]//2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def _rhat(x):
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1))
    if within <= 0:
        return np.inf
    between = n*np.var(np.mean(x, axis=1), ddof=1)
    return float(np.sqrt(((n-1)*within/n+between/n)/within))


def _ess(x):
    """Multi-chain FFT autocovariances and Geyer initial positive/monotone pairs."""
    chains, n = x.shape
    if n < 4:
        return 0.
    scale = np.max(np.abs(x))
    if scale == 0:
        return 0.
    x = x/scale  # ESS is scale invariant; avoid overflow in autocovariances
    centered = x-x.mean(axis=1, keepdims=True)
    size = 1 << (2*n-1).bit_length()
    fft = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(fft*np.conjugate(fft), n=size, axis=1)[:, :n]/n
    within = float(np.mean(acov[:, 0])*n/(n-1))
    varplus = (n-1)*within/n+np.var(x.mean(axis=1), ddof=1)
    if varplus <= 0 or within <= 0:
        return 0.
    rho = 1-(within-acov.mean(axis=0))/varplus
    rho[0] = 1
    pairs = []
    for j in range(0, n-1, 2):
        pair = rho[j]+rho[j+1]
        if pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(-1+2*sum(pairs), 1/np.log10(max(chains*n, 10)))
    return float(min(chains*n/tau, chains*n))


def chain_diagnostics(draws):
    """Rank-normalized split/folded Rhat, bulk/tail ESS and raw-mean MCSE.

    MCSE here assumes finite marginal variance; callers must enforce moment
    conditions before presenting it as a posterior-mean MCSE.
    """
    x = np.asarray(draws, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.all(np.isfinite(x)):
        raise ValueError('require >=2 chains, >=8 retained draws, finite values')
    split = _split(x)
    def ranks(v):
        r = rankdata(v.ravel(), method='average').reshape(v.shape)
        return norm.ppf((r-.375)/(v.size+.25))
    ranked = ranks(split)
    folded = ranks(np.abs(split-np.median(split)))
    bulk = _ess(ranked)
    qlo, qhi = np.quantile(x, [.05, .95])
    tail = min(_ess((split <= qlo).astype(float)), _ess((split <= qhi).astype(float)))
    mean_ess = _ess(split)
    scale = np.max(np.abs(x))
    mcse = (float(np.std(x/scale, ddof=1)*(scale/np.sqrt(mean_ess)))
            if mean_ess and scale else None)
    return dict(rhat=max(_rhat(ranked), _rhat(folded)), ess_bulk=bulk,
                ess_tail=tail, ess_mean=mean_ess, mcse_mean=mcse)
