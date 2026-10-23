"""Run from the repository root: python -m experiments.stage2_example.

This is a deterministic teaching example, not a BPCS simulator or estimator.
"""

import json
from pathlib import Path

from src.censoring import BPCSExperiment, BlockData, BlockScheme, Observation
from src.model import CompetingRisks


def load_example(path=None):
    """Load this example's JSON layout through the shared validating constructors."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data/examples/two_blocks.json"
    payload = json.loads(Path(path).read_text())
    return BPCSExperiment(tuple(
        BlockData(BlockScheme(**b["scheme"]), tuple(Observation(**r) for r in b["observations"]))
        for b in payload["blocks"]
    ))


def main():
    data = load_example()
    print("Hand-constructed teaching data; no simulation or estimation.\n")
    print(data.to_frame().to_string(index=False))
    summary = data.summary_frame()
    print("\nBlock accounting:")
    print(summary.to_string(index=False))
    print(f"\nTotals: n={summary.n.sum()}, failures={summary.m.sum()}, "
          f"withdrawn={summary.withdrawn.sum()}")

    # Chosen values illustrate the existing Stage 1 layer, not inferred parameters.
    print("\nIllustrative Stage 1 reliability at t=0.5, with shared a=2:")
    for label, b1, b2 in [("A", 1.0, 2.0), ("B", 0.8, 1.2)]:
        model = CompetingRisks(a=2.0, b1=b1, b2=b2)
        print(f"{label}: b1={b1}, b2={b2}, S(0.5)={model.minimum.sf(.5):.6f}, "
              f"P(cause 1)={model.cause_probability(1):.6f}")


if __name__ == "__main__":
    main()
