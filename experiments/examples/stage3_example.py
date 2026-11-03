"""Fixed-parameter likelihood evaluation of Stage 2 teaching data, without fitting.

Run from the project root: python -m experiments.examples.stage3_example.
"""

from experiments.examples.stage2_example import load_example
from src.likelihood import block_loglikelihood, log_censoring_constant, loglikelihood
from src.model import CompetingRisks


def main():
    data = load_example()
    a = 2.0
    b = {"A": (1.0, 2.0), "B": (0.8, 1.2)}
    print("Hand-constructed teaching data; chosen parameters, no estimation.")
    print("Shared a=2; A: (b1,b2)=(1,2); B: (b1,b2)=(0.8,1.2)\n")
    print("block j time cause r   log_failure log_withdrawals    log_row")
    for block in data.blocks:
        b1, b2 = b[block.scheme.block]
        model = CompetingRisks(a, b1, b2)
        for j, row in enumerate(block.observations, 1):
            failure = model.cause_logpdf(row.time, row.cause)
            withdrawal = row.withdrawals * model.minimum.logsf(row.time)
            print(f"{row.block:>5} {j} {row.time:.2f} {row.cause:>5} {row.withdrawals} "
                  f"{failure:13.9f} {withdrawal:15.9f} {failure+withdrawal:12.9f}")
        kernel = block_loglikelihood(block, a, b1, b2, include_constant=False)
        logc = log_censoring_constant(block)
        full = block_loglikelihood(block, a, b1, b2)
        print(f"  {block.scheme.block}: log kernel={kernel:.12f}, "
              f"log C={logc:.12f}, full log L={full:.12f}\n")
    print(f"Joint log kernel: {loglikelihood(data, a, b, include_constant=False):.12f}")
    print(f"Joint log C:      {log_censoring_constant(data):.12f}")
    print(f"Joint full log L: {loglikelihood(data, a, b):.12f}")


if __name__ == "__main__":
    main()
