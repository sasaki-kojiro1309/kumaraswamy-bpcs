"""Validated, completed BPCS data on Kumaraswamy support (0,1).

Reference 1 — Wang et al. (2023), Section 2.1, Eq. (4), supplies the
experimental organization. No likelihood, random withdrawals, or fitting is
performed here. Withdrawal selection must be noninformative in the experiment;
counts alone cannot establish that assumption.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _block_id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("block must be a nonempty string")
    return value


@dataclass(frozen=True)
class BlockScheme:
    """Prefixed design: block ID, initial n, failure target m, withdrawal tuple.

    Each withdrawal count applies immediately AFTER its corresponding failure.
    Complete observations (m=n, all withdrawals zero) are allowed.
    """

    block: str
    n: int
    m: int
    withdrawals: tuple[int, ...]

    def __post_init__(self):
        _block_id(self.block)
        object.__setattr__(self, "n", _integer("n", self.n, 1))
        object.__setattr__(self, "m", _integer("m", self.m, 1))
        if self.m > self.n:
            raise ValueError("m cannot exceed n")
        try:
            counts = tuple(_integer("withdrawal count", r) for r in self.withdrawals)
        except TypeError as exc:
            raise ValueError("withdrawals must be an iterable of integer counts") from exc
        object.__setattr__(self, "withdrawals", counts)
        if len(counts) != self.m:
            raise ValueError("withdrawals must have exactly m entries")
        remaining = self.n
        for j, r in enumerate(counts, start=1):
            if remaining < 1:
                raise ValueError(f"block {self.block}, failure {j}: no unit remains to fail")
            if r > remaining - 1:
                raise ValueError(f"block {self.block}, failure {j}: withdrawals exceed survivors")
            remaining -= 1 + r
        if remaining != 0:
            raise ValueError("completed block requires n = m + sum(withdrawals)")

    @property
    def risk_sets(self):
        """Numbers alive and under test immediately before each observed failure."""
        remaining = self.n
        risks = []
        for r in self.withdrawals:
            risks.append(remaining)
            remaining -= 1 + r
        return tuple(risks)


@dataclass(frozen=True)
class Observation:
    """One failure, its cause, and the survivors withdrawn at that time.

    None is the reserved representation of an unknown cause. It is representable
    at row level but rejected by the current completed-block baseline.
    """

    block: str
    time: float
    cause: int | None
    withdrawals: int

    def __post_init__(self):
        _block_id(self.block)
        if (isinstance(self.time, bool) or not isinstance(self.time, Real)
                or not isfinite(self.time) or not 0 < self.time < 1):
            raise ValueError("failure time must be finite and strictly inside (0,1)")
        object.__setattr__(self, "time", float(self.time))
        if self.cause is not None:
            cause = _integer("cause", self.cause, 1)
            if cause not in (1, 2):
                raise ValueError("cause must be 1, 2, or None (reserved)")
            object.__setattr__(self, "cause", cause)
        object.__setattr__(self, "withdrawals", _integer("withdrawals", self.withdrawals))


@dataclass(frozen=True)
class BlockData:
    """A completed block matching its design; all causes must be known.

    Observations must arrive in within-block failure order. They are never
    silently sorted, because that could hide a schedule-to-failure mismatch.
    """

    scheme: BlockScheme
    observations: tuple[Observation, ...]

    def __post_init__(self):
        if not isinstance(self.scheme, BlockScheme):
            raise ValueError("scheme must be a BlockScheme")
        try:
            rows = tuple(self.observations)
        except TypeError as exc:
            raise ValueError("observations must be an iterable of Observation objects") from exc
        object.__setattr__(self, "observations", rows)
        if len(rows) != self.scheme.m:
            raise ValueError("completed block must contain exactly m observed failures")
        previous_time = 0.0
        for j, (row, r) in enumerate(zip(rows, self.scheme.withdrawals), start=1):
            if not isinstance(row, Observation):
                raise ValueError("each row must be an Observation")
            context = f"block {self.scheme.block}, failure {j}"
            if row.block != self.scheme.block:
                raise ValueError(f"{context}: observation has a different block ID")
            if row.cause is None:
                raise ValueError(f"{context}: unknown causes are not supported in the baseline")
            if row.withdrawals != r:
                raise ValueError(f"{context}: observed withdrawals do not match the scheme")
            if row.time <= previous_time:
                raise ValueError(f"{context}: times must be strictly increasing within a block")
            previous_time = row.time

    @property
    def cause_counts(self):
        """Numbers of observed failures of causes 1 and 2 (not latent causes)."""
        return {c: sum(row.cause == c for row in self.observations) for c in (1, 2)}


@dataclass(frozen=True)
class BPCSExperiment:
    """Any positive number of completed, uniquely named testing blocks.

    Times need only be ordered WITHIN each block. All-cause-1 or all-cause-2
    blocks remain valid data; future estimation must handle their boundaries.
    """

    blocks: tuple[BlockData, ...]

    def __post_init__(self):
        try:
            blocks = tuple(self.blocks)
        except TypeError as exc:
            raise ValueError("blocks must be an iterable of BlockData objects") from exc
        object.__setattr__(self, "blocks", blocks)
        if not blocks:
            raise ValueError("experiment must contain at least one block")
        if not all(isinstance(b, BlockData) for b in blocks):
            raise ValueError("each block must be a BlockData")
        ids = [b.scheme.block for b in blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("block IDs must be unique")

    def to_frame(self):
        """Return a fresh observation/risk-set table; editing it cannot mutate data."""
        import pandas as pd

        return pd.DataFrame([
            {"block": block.scheme.block, "failure_index": j,
             "time": row.time, "cause": row.cause, "withdrawals": row.withdrawals,
             "at_risk": risk, "after_failure": risk - 1,
             "remaining": risk - 1 - row.withdrawals}
            for block in self.blocks
            for j, (row, risk) in enumerate(
                zip(block.observations, block.scheme.risk_sets), start=1)
        ])

    def summary_frame(self):
        """Return per-block design sizes, withdrawals, and observed cause counts."""
        import pandas as pd

        return pd.DataFrame([
            {"block": b.scheme.block, "n": b.scheme.n, "m": b.scheme.m,
             "withdrawn": sum(b.scheme.withdrawals),
             "cause_1": b.cause_counts[1], "cause_2": b.cause_counts[2]}
            for b in self.blocks
        ])
