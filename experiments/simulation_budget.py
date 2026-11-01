"""POSIX elapsed-time supervision for fits, including in-flight MCMC calls."""
from contextlib import contextmanager
import math
import signal
import time


class BudgetExpired(BaseException):
    """Not an estimator error: bypass broad Exception handlers in adapters."""


class ExecutionBudget:
    def __init__(self, seconds=None, deadline_unix=None, reserve_seconds=2.):
        now=time.monotonic()
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError('budget seconds must be finite and nonnegative')
        if deadline_unix is not None and not math.isfinite(deadline_unix):
            raise ValueError('deadline must be finite')
        limits=[now+seconds] if seconds is not None else []
        if deadline_unix is not None:
            limits.append(now+max(0.,deadline_unix-time.time()))
        self.deadline=min(limits) if limits else math.inf
        self.reserve_seconds=reserve_seconds

    def remaining(self):
        return max(0.,self.deadline-time.monotonic()-self.reserve_seconds)

    @contextmanager
    def supervise(self):
        remaining=self.remaining()
        if remaining<=0:
            raise BudgetExpired()
        if math.isinf(remaining):
            yield
            return
        if not hasattr(signal,'setitimer'):
            raise RuntimeError('Hard fit supervision requires POSIX setitimer; use an external supervisor on this platform')
        def expire(signum,frame): raise BudgetExpired()
        old_handler=signal.signal(signal.SIGALRM,expire)
        old_timer=signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.)
            signal.signal(signal.SIGALRM,old_handler)
            if old_timer[0]>0:
                signal.setitimer(signal.ITIMER_REAL,old_timer[0],old_timer[1])
