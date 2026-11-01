"""Bayesian Kumaraswamy BPCS: independent scale prior and hierarchical Gamma.

Exactly two causes, arbitrary blocks, optional WITHIN-block b1<=b2 only.
All Gamma distributions use shape-rate. See BAYESIAN_DERIVATIONS.md.
"""
from dataclasses import dataclass, asdict
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.special import gammaln, betainc, betaincinv, logsumexp

from .likelihood import loglikelihood
from .mle import _Problem, _positive
from .model import Kumaraswamy
from .bayesian_sampling import (SamplingError, slice_step, truncated_gamma,
                                shortest_interval, chain_diagnostics)


class PosteriorError(ValueError):
    """Improper posterior, or a model outside the explicitly supported domain."""


@dataclass(frozen=True)
class GammaPrior:
    shape: float
    rate: float

    def __post_init__(self):
        object.__setattr__(self, 'shape', _positive(self.shape, 'Gamma shape'))
        object.__setattr__(self, 'rate', _positive(self.rate, 'Gamma rate'))

    def logpdf(self, value):
        value = _positive(value, 'Gamma value')
        return float(self.shape*np.log(self.rate)-gammaln(self.shape)
                     +(self.shape-1)*np.log(value)-self.rate*value)


@dataclass(frozen=True)
class HierarchicalPrior:
    """Required explicit hyperpriors; tuples list cause 1 then cause 2."""
    a: GammaPrior
    kappa: tuple[GammaPrior, GammaPrior]
    lam: tuple[GammaPrior, GammaPrior]

    def __post_init__(self):
        if not isinstance(self.a, GammaPrior):
            raise ValueError('a must have a GammaPrior')
        for name in ('kappa', 'lam'):
            values = tuple(getattr(self, name))
            if len(values) != 2 or not all(isinstance(p, GammaPrior) for p in values):
                raise ValueError(name+' requires exactly two GammaPrior objects')
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class SamplingConfig:
    chains: int = 4
    warmup: int = 1000
    draws: int = 2000
    seed: int = 20261101
    slice_width: float = 1.
    max_slice_steps: int = 1000
    max_rhat: float = 1.01
    min_ess: float = 400.
    probability: float = .95

    def __post_init__(self):
        for key, minimum in [('chains', 2), ('warmup', 0), ('draws', 8),
                             ('seed', 0), ('max_slice_steps', 1)]:
            v = getattr(self, key)
            if not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) or v < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        for key in ('slice_width', 'max_rhat', 'min_ess', 'probability'):
            _positive(getattr(self, key), key)
        if self.max_rhat <= 1 or self.probability >= 1:
            raise ValueError('require max_rhat>1 and probability<1')


@dataclass
class BayesianResult:
    method: str
    restricted: bool
    status: str
    converged: bool
    names: tuple
    draws: np.ndarray  # chain, retained iteration, parameter
    estimates: dict
    intervals: dict
    moments: dict
    diagnostics: dict
    warnings: list
    reliability: dict
    configuration: dict


class _Context:
    def __init__(self, data):
        self.problem = _Problem(data)
        self.data = data
        self.labels = tuple(row[0] for row in self.problem.blocks)
        self.counts = np.array([row[4] for row in self.problem.blocks])
        self.m = self.counts.sum(axis=1)
        self.ymin = np.array([row[2].min() for row in self.problem.blocks])
        differences = []
        for _, t, _, _, _ in self.problem.blocks:
            delta = np.log(t[-1])-np.log(t)
            close = t > t[-1]/2
            delta[close] = np.log1p((t[-1]-t[close])/t[close])
            differences.append(float(delta.sum()))
        self.D = float(sum(differences))
        self.names = ('a',)+tuple(f'b{c}[{s}]' for s in self.labels for c in (1, 2))

    def pairs(self, b):
        b = np.asarray(b, dtype=float)
        if b.shape != self.counts.shape or not np.all(np.isfinite(b)) or np.any(b <= 0):
            raise ValueError('b must be a finite positive (number of blocks,2) array')
        return dict(zip(self.labels, map(tuple, b)))

    def summaries(self, a):
        a = _positive(a, 'a')
        logW, H = [], []
        for _, t, u, w, _ in self.problem.blocks:
            z = a*u
            L = np.asarray(Kumaraswamy(a, 1).logsf(t))
            lv = np.empty_like(z); tail = z > 36
            lv[tail] = -z[tail]
            lv[~tail] = np.log(-L[~tail])
            logW.append(logsumexp(np.log(w)+lv)); H.append(L.sum())
        if not np.all(np.isfinite([*logW, *H])):
            raise SamplingError('Bayesian summaries outside float64 range')
        return np.array(logW), np.array(H)


def posterior_conditions(data, *, prior='independent', restricted=False):
    """Propriety and sufficient first/second-moment checks, never from MC means.

    Independent zero-count cases are conservatively refused in BOTH models;
    some restricted zero-cause-2 cases can in fact be proper (documented).
    """
    if not isinstance(restricted, bool):
        raise ValueError('restricted must be boolean')
    ctx = _Context(data)
    independent = isinstance(prior, str) and prior == 'independent'
    if not independent and not isinstance(prior, HierarchicalPrior):
        raise ValueError('prior must be independent or HierarchicalPrior')
    counts_ok = bool(np.all(ctx.counts > 0))
    proper = (counts_ok and ctx.problem.M > 1 and ctx.D > 0) if independent else True
    if independent and restricted and not counts_ok:
        proper = None  # conservatively unsupported; not a claim of impropriety
    moments = {'a': {'first': 'finite', 'second': 'finite'}}
    penalty = 0 if independent else prior.a.rate
    for i, label in enumerate(ctx.labels):
        for c in (1, 2):
            status = {}
            for order, key in [(1, 'first'), (2, 'second')]:
                status[key] = ('finite' if penalty+ctx.D > order*ctx.ymin[i] else
                               'infinite' if independent and proper else 'unverified')
            moments[f'b{c}[{label}]'] = status
    if not independent:
        moments.update({f'{name}{s}': {'first': 'finite', 'second': 'finite'}
                        for name in ('kappa', 'lambda') for s in (1, 2)})
    if proper is not True:
        moments = {name: dict(first='undefined' if proper is False else 'unverified',
                              second='undefined' if proper is False else 'unverified')
                   for name in moments}
    return dict(proper=proper, status='proper' if proper else
                'unsupported_zero_count' if restricted and not counts_ok else 'improper_posterior',
                all_cause_counts_positive=counts_ok, M=ctx.problem.M, D=ctx.D,
                y_min=dict(zip(ctx.labels, ctx.ymin)), moments=moments,
                note='Hierarchical b moment conditions are sufficient, not necessary. '
                     'Independent zero counts are conservatively excluded even under restriction.')


def independent_logposterior(data, a, b_by_block, *, restricted=False):
    """Unnormalized posterior in ORIGINAL coordinates, including the full log C."""
    value = loglikelihood(data, a, b_by_block)
    if restricted and any(pair[0] > pair[1] for pair in b_by_block.values()):
        return -np.inf
    return float(value-np.log(a)-sum(np.log(pair).sum() for pair in b_by_block.values()))


def independent_log_marginal(data, a):
    """Log h_I(a), dropping only a-independent Gamma/count and C constants."""
    ctx = _Context(data)
    logW, H = ctx.summaries(a)
    return float((ctx.problem.M-1)*np.log(a)+(a-1)*ctx.problem.T-H.sum()-ctx.m@logW)


def independent_conditionals(data, a):
    """Conditional Gamma shape/rate and Beta shapes; refuses zero-count priors."""
    check = posterior_conditions(data)
    if not check['proper']:
        raise PosteriorError(str(check))
    ctx = _Context(data); logW, _ = ctx.summaries(a)
    return {s: dict(cause_shapes=tuple(ctx.counts[i]), log_rate=float(logW[i]),
                    total_shape=float(ctx.m[i]), beta_shapes=tuple(ctx.counts[i]))
            for i, s in enumerate(ctx.labels)}


def hierarchical_logposterior(data, a, b_by_block, kappa, lam, prior, *, restricted=False):
    """Joint kernel including ALL normalizers depending on unknown parameters.

    Fixed hyperprior normalizers and log C are included too; the conditioned
    restricted model multiplies this by the event indicator, with no state-
    dependent prior-event normalization.
    """
    if not isinstance(prior, HierarchicalPrior):
        raise ValueError('HierarchicalPrior required')
    ctx = _Context(data)
    b = np.array([b_by_block[s] for s in ctx.labels], dtype=float)
    ctx.pairs(b)
    kappa, lam = _hyper(kappa, 'kappa'), _hyper(lam, 'lambda')
    value = loglikelihood(data, a, b_by_block)
    if restricted and np.any(b[:, 0] > b[:, 1]):
        return -np.inf
    value += prior.a.logpdf(a)
    for s in range(2):
        value += prior.kappa[s].logpdf(kappa[s])+prior.lam[s].logpdf(lam[s])
        value += len(b)*(kappa[s]*np.log(lam[s])-gammaln(kappa[s]))
        value += (kappa[s]-1)*np.log(b[:, s]).sum()-lam[s]*b[:, s].sum()
    if not np.isfinite(value):
        raise SamplingError('joint posterior outside numerical range')
    return float(value)


def _hyper(values, name):
    v = np.asarray(values, dtype=float)
    if v.shape != (2,) or not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError(name+' requires two finite positive values')
    return v


def hierarchical_conditionals(data, a, b_by_block, kappa, lam, prior):
    """Standard Gamma conditional parameters, shape-rate (rates on log scale)."""
    ctx = _Context(data); b = np.array([b_by_block[s] for s in ctx.labels])
    ctx.pairs(b); kappa, lam = _hyper(kappa, 'kappa'), _hyper(lam, 'lambda')
    if not isinstance(prior, HierarchicalPrior):
        raise ValueError('HierarchicalPrior required')
    logW, _ = ctx.summaries(a)
    return dict(b_shape=ctx.counts+kappa,
                b_lograte=np.logaddexp(logW[:, None], np.log(lam)[None, :]),
                lambda_shape=len(b)*kappa+np.array([p.shape for p in prior.lam]),
                lambda_rate=b.sum(axis=0)+np.array([p.rate for p in prior.lam]))


def kappa_logconditional(value, b_column, lam, prior):
    """Nonstandard shape conditional in the raw positive kappa coordinate."""
    value = _positive(value, 'kappa'); lam = _positive(lam, 'lambda')
    b = np.asarray(b_column, dtype=float)
    if b.ndim != 1 or not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError('positive b column required')
    return float((prior.shape-1)*np.log(value)-prior.rate*value
                 +value*(len(b)*np.log(lam)+np.log(b).sum())-len(b)*gammaln(value))


def a_logconditional(data, a, b_by_block, prior):
    """Nonstandard common-shape conditional in a; b held fixed."""
    return loglikelihood(data, a, b_by_block)+prior.a.logpdf(a)


def _gamma_draw(shape, lograte, rng):
    """Draw via a log unit Gamma; tiny shapes use the exact boosting identity."""
    if shape < 1:
        logunit = np.log(rng.gamma(shape+1))-rng.exponential()/shape
    else:
        logunit = np.log(rng.gamma(shape))
    logvalue = logunit-lograte
    if not -744 < logvalue < 709:
        raise SamplingError('positive Gamma draw outside representable float64 range')
    value = float(np.exp(logvalue))
    if not np.isfinite(value) or value <= 0:
        raise SamplingError('nonpositive or nonfinite Gamma draw')
    return value


def _positive_log_target(log_value, raw_target):
    if not -700 < log_value < 700:
        raise SamplingError('slice proposal exceeds supported log-parameter range; no clipping applied')
    value = float(np.exp(log_value))
    result = raw_target(value)+log_value  # Jacobian dv/dlog(v)=v
    if not np.isfinite(result):
        raise SamplingError('nonfinite transformed conditional')
    return float(result)


def _independent_chain(ctx, config, seed, restricted):
    streams = [np.random.default_rng(s) for s in seed.spawn(3)]
    ra, rb, rp = streams
    eta = float(np.log(ctx.problem.M/-ctx.problem.T)+ra.normal())
    out = np.empty((config.draws, len(ctx.names)))
    def target(z):
        return _positive_log_target(z, lambda a: _marginal(ctx, a))
    for j in range(config.warmup+config.draws):
        eta = slice_step(target, eta, ra, width=config.slice_width, max_steps=config.max_slice_steps)
        if j < config.warmup:
            continue
        a = float(np.exp(eta)); logW, _ = ctx.summaries(a)
        b = np.empty_like(ctx.counts)
        for i, d in enumerate(ctx.counts):
            B = _gamma_draw(ctx.m[i], logW[i], rb)
            if restricted:
                mass = betainc(*d, .5)
                if not mass > 0:
                    raise SamplingError('truncated Beta event probability underflows')
                p = float(betaincinv(*d, mass*rp.random()))
            else:
                p = float(rp.beta(*d))
            b[i] = B*p, B*(1-p)
        if np.any(b <= 0) or not np.all(np.isfinite(b)):
            raise SamplingError('Beta/Gamma reconstruction outside positive float64 support')
        out[j-config.warmup] = [a, *b.ravel()]
    return out


def _marginal(ctx, a):
    logW, H = ctx.summaries(a)
    return float((ctx.problem.M-1)*np.log(a)+(a-1)*ctx.problem.T-H.sum()-ctx.m@logW)


def _hierarchical_chain(ctx, prior, config, seed, restricted):
    rng = np.random.default_rng(seed)
    def initial(p, width):
        eta = np.log(p.shape)-np.log(p.rate)+rng.normal(0, width)
        if not -700 < eta < 700:
            raise SamplingError('prior-mean initialization outside supported numerical range')
        return float(np.exp(eta))
    a = initial(prior.a, .5)
    kap = np.array([initial(p, .3) for p in prior.kappa])
    lam = np.array([initial(p, .3) for p in prior.lam])
    b = np.exp(rng.normal(0, .3, ctx.counts.shape))
    if restricted:
        b[:, 1] += b[:, 0]  # initialization only, not projection of posterior draws
    out = np.empty((config.draws, len(ctx.names)+4))
    for j in range(config.warmup+config.draws):
        logW, _ = ctx.summaries(a)
        for i in range(len(b)):
            for s in range(2):
                shape = kap[s]+ctx.counts[i, s]
                logr = np.logaddexp(np.log(lam[s]), logW[i])
                if restricted:
                    rate = float(np.exp(logr))
                    b[i, s] = truncated_gamma(shape, rate, rng,
                        lower=0. if s == 0 else b[i, 0], upper=b[i, 1] if s == 0 else np.inf)
                else:
                    b[i, s] = _gamma_draw(shape, logr, rng)
        for s in range(2):
            lam[s] = _gamma_draw(len(b)*kap[s]+prior.lam[s].shape,
                                  np.log(prior.lam[s].rate+b[:, s].sum()), rng)
            def target(z):
                return _positive_log_target(z, lambda k: kappa_logconditional(k, b[:, s], lam[s], prior.kappa[s]))
            kap[s] = np.exp(slice_step(target, np.log(kap[s]), rng,
                        width=config.slice_width, max_steps=config.max_slice_steps))
        pairs = ctx.pairs(b)
        def target_a(z):
            return _positive_log_target(z, lambda v: loglikelihood(ctx.data, v, pairs)+prior.a.logpdf(v))
        a = np.exp(slice_step(target_a, np.log(a), rng, width=config.slice_width,
                              max_steps=config.max_slice_steps))
        if j >= config.warmup:
            out[j-config.warmup] = [a, *b.ravel(), *kap, *lam]
    return out


def _summary(x, moment, config):
    diag = chain_diagnostics(x)
    if moment['second'] != 'finite':
        diag['mcse_mean'] = None
        diag['mcse_note'] = 'finite second moment not established'
    passed = bool(diag['rhat'] <= config.max_rhat and
                  min(diag['ess_bulk'], diag['ess_tail']) >= config.min_ess)
    scale = np.max(np.abs(x))
    estimate = (float(scale*np.mean(x/scale)) if scale else 0.) if moment['first'] == 'finite' and passed else None
    return estimate, shortest_interval(x, config.probability), diag, passed


def posterior_reliability(draws, names, labels, times):
    """Evaluate each joint draw, using Stage 1 models; never plug in mean parameters."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any((times <= 0) | (times >= 1)):
        raise ValueError('mission times must be finite in (0,1)')
    index = {s: j for j, s in enumerate(names)}
    arrays = {}
    for label in labels:
        a = draws[:, :, index['a']]
        b1 = draws[:, :, index[f'b1[{label}]']]; b2 = draws[:, :, index[f'b2[{label}]']]
        B = b1+b2; p = b1/B
        block = {'cause_1_probability': p, 'cause_2_probability': 1-p}
        # a and B vary, whereas Stage 1 parameters are scalar. Iterate draws,
        # keeping its tested point formulas instead of duplicating them.
        med = np.empty_like(a)
        for pos in np.ndindex(a.shape):
            med[pos] = Kumaraswamy(a[pos], B[pos]).median()
        block['median'] = med
        for t in times:
            suffix = repr(float(t))
            S = np.empty_like(a); h = np.empty_like(a); F = np.empty_like(a)
            for pos in np.ndindex(a.shape):
                model = Kumaraswamy(a[pos], B[pos])
                S[pos], h[pos], F[pos] = model.sf(t), model.hazard(t), model.cdf(t)
            block[f'survival@{suffix}'] = S; block[f'hazard@{suffix}'] = h
            block[f'cause_1_incidence@{suffix}'] = p*F
            block[f'cause_2_incidence@{suffix}'] = (1-p)*F
        arrays[label] = block
    return arrays


def fit(data, *, prior, restricted=False, config=None, times=(.5,)):
    """Run any of the four requested methods on the common BPCS data interface.

    posterior_conditions refuses unsupported independent priors before sampling.
    Diagnostics are necessary checks, not proof of convergence. Failed checks
    withhold estimates; raw draws/intervals remain explicitly exploratory.
    Infinite/unverified first moments NEVER yield squared-error Bayes estimates.
    """
    config = SamplingConfig() if config is None else config
    if not isinstance(config, SamplingConfig):
        raise ValueError('config must be SamplingConfig')
    # Validate time arguments even before a potentially long sampler run.
    tt = np.atleast_1d(np.asarray(times, dtype=float))
    if tt.ndim != 1 or not np.all(np.isfinite(tt)) or np.any((tt <= 0) | (tt >= 1)):
        raise ValueError('mission times must lie in (0,1)')
    check = posterior_conditions(data, prior=prior, restricted=restricted)
    if not check['proper']:
        raise PosteriorError(str(check))
    ctx = _Context(data); independent = isinstance(prior, str)
    names = ctx.names if independent else ctx.names+('kappa1', 'kappa2', 'lambda1', 'lambda2')
    method = 'independent' if independent else 'hierarchical_gamma'
    start = perf_counter(); chains = []; warnings = []
    seeds = np.random.SeedSequence(config.seed).spawn(config.chains)
    try:
        for seed in seeds:
            with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
                chain = (_independent_chain(ctx, config, seed, restricted) if independent else
                         _hierarchical_chain(ctx, prior, config, seed, restricted))
            if not np.all(np.isfinite(chain)) or np.any(chain <= 0):
                raise SamplingError('nonfinite/nonpositive retained state')
            chains.append(chain)
    except (SamplingError, FloatingPointError, OverflowError) as exc:
        return BayesianResult(method, restricted, 'numerical_failure', False, names,
            np.empty((0, 0, len(names))), {}, {}, check['moments'],
            dict(error=str(exc), completed_chains=len(chains), elapsed_seconds=perf_counter()-start),
            [str(exc)], {}, dict(prior=asdict(prior) if not independent else prior, sampler=asdict(config)))
    draws = np.stack(chains)
    estimates, intervals, diagnostics = {}, {}, {}
    success = True
    for j, name in enumerate(names):
        estimate, interval, diag, passed = _summary(draws[:, :, j], check['moments'][name], config)
        estimates[name], intervals[name], diagnostics[name] = estimate, interval, diag
        success &= passed
        if check['moments'][name]['first'] != 'finite':
            warnings.append(name+': posterior first moment '+check['moments'][name]['first']+'; mean withheld')
        elif check['moments'][name]['second'] != 'finite':
            warnings.append(name+': finite variance not established; mean MCSE unavailable')
    reliability = {}
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            values = posterior_reliability(draws, names, ctx.labels, tt)
        if not all(np.all(np.isfinite(x)) for entries in values.values() for x in entries.values()):
            raise SamplingError('nonfinite draw-wise reliability value')
    except (SamplingError, FloatingPointError, OverflowError) as exc:
        return BayesianResult(method, restricted, 'numerical_failure', False, names, draws,
            dict.fromkeys(names), {}, check['moments'],
            dict(error=str(exc), parameters=diagnostics, elapsed_seconds=perf_counter()-start),
            warnings+[str(exc)], {},
            dict(prior=asdict(prior) if not independent else prior, sampler=asdict(config)))
    for i, (label, entries) in enumerate(values.items()):
        block = {}
        for name, x in entries.items():
            moment = dict(first='finite', second='finite')
            if name.startswith('hazard'):
                t = float(name.split('@')[1])
                gap = ctx.ymin[i]+np.log(t)
                for order, key in [(1, 'first'), (2, 'second')]:
                    margin = ctx.D+(0 if independent else prior.a.rate)-order*gap
                    moment[key] = ('finite' if margin > 0 else
                                   'infinite' if independent else 'unverified')
            estimate, interval, diag, passed = _summary(x, moment, config)
            block[name] = dict(estimate=estimate, interval=interval, diagnostics=diag,
                               moments=moment, draws=x)
            if moment['first'] != 'finite':
                warnings.append(label+'/'+name+': first moment '+moment['first']+'; mean withheld')
            elif moment['second'] != 'finite':
                warnings.append(label+'/'+name+': finite variance not established; mean MCSE unavailable')
            success &= passed
        reliability[label] = block
    if not success:
        warnings.append('Convergence diagnostics failed; all Bayes estimates withheld. Intervals/draws are exploratory.')
        estimates = dict.fromkeys(names)
        for entries in reliability.values():
            for entry in entries.values(): entry['estimate'] = None
    return BayesianResult(method, restricted, 'diagnostics_passed' if success else 'diagnostics_failed',
        bool(success), names, draws, estimates, intervals, check['moments'],
        dict(parameters=diagnostics, propriety=check, elapsed_seconds=perf_counter()-start,
             criteria=dict(max_rhat=config.max_rhat, min_ess=config.min_ess),
             note='Passing rank Rhat/ESS checks does not prove convergence.'), warnings, reliability,
        dict(prior=asdict(prior) if not independent else prior, sampler=asdict(config)))
