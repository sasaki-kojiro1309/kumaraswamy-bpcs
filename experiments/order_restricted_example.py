"""Compare fits to unchanged teaching data; no simulation or uncertainty.

Run from the repository root: python -m experiments.order_restricted_example
"""

from experiments.stage2_example import load_example
from src.mle import fit_mle
from src.order_restricted import fit_order_restricted_mle


def main():
    data = load_example()  # data/examples/two_blocks.json
    unrestricted = fit_mle(data)
    restricted = fit_order_restricted_mle(data)
    for name, result in (("Unrestricted", unrestricted), ("Restricted", restricted)):
        print(f"{name}: status={result.status}, converged={result.converged}")
        if not result.converged:
            print(result.message)
            return
        print(f"  a={result.a:.12f}; full log-likelihood={result.loglikelihood:.12f}")
        for label, (b1, b2) in result.b_by_block.items():
            print(f"  {label}: b1={b1:.12f}, b2={b2:.12f}, B={b1+b2:.12f}")
    print("Active equalities:", restricted.diagnostics['active_constraints'])
    print("Zero-parameter boundaries:", restricted.diagnostics['boundary_parameters'])
    print(f"Log-likelihood penalty: {unrestricted.loglikelihood-restricted.loglikelihood:.12f}")
    print("The order is provisional; these are hand-constructed teaching data.")


if __name__ == '__main__':
    main()
