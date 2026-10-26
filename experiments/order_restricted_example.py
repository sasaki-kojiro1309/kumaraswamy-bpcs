"""Compare fits to unchanged teaching data; no simulation or uncertainty.

Run from the repository root: python -m experiments.order_restricted_example
"""

from experiments.stage2_example import load_example
from experiments.within_block_validation import fit_slsqp
from src.mle import fit_mle
from src.model import Kumaraswamy
from src.order_restricted import fit_order_restricted_mle


def main():
    data = load_example()  # data/examples/two_blocks.json
    print("Input: hand-constructed block data, with fixed withdrawal schemes")
    print(data.to_frame().to_string(index=False))
    print()
    unrestricted = fit_mle(data)
    restricted = fit_order_restricted_mle(data)
    numerical = fit_slsqp(data)
    for name, result in (("Unrestricted", unrestricted), ("Restricted", restricted),
                         ("Independent joint SLSQP", numerical)):
        print(f"{name}: status={result.status}, converged={result.converged}")
        if not result.converged:
            print(result.message)
            return
        print(f"  a={result.a:.12f}; full log-likelihood={result.loglikelihood:.12f}")
        for label, (b1, b2) in result.b_by_block.items():
            print(f"  {label}: b1={b1:.12f}, b2={b2:.12f}, B={b1+b2:.12f}, "
                  f"P(cause 1)={b1/(b1+b2):.12f}")
        if name == "Independent joint SLSQP":
            print("  Solver checks:", result.diagnostics['starts'])
    print("Active equalities:", restricted.diagnostics['active_constraints'])
    print("Strict violations pooled:", restricted.diagnostics['pooled_blocks'])
    print("Zero-parameter boundaries:", restricted.diagnostics['boundary_parameters'])
    print(f"Log-likelihood penalty: {unrestricted.loglikelihood-restricted.loglikelihood:.12f}")
    print(f"SLSQP/analytical log-likelihood difference: {abs(numerical.loglikelihood-restricted.loglikelihood):.3e}")
    for b in data.blocks:
        label = b.scheme.block
        u_pair, r_pair = unrestricted.b_by_block[label], restricted.b_by_block[label]
        u_survival = Kumaraswamy(unrestricted.a, sum(u_pair)).sf(.5)
        r_survival = Kumaraswamy(restricted.a, sum(r_pair)).sf(.5)
        print(f"{label}: minimum-lifetime S(.5) unrestricted={u_survival:.12f}, "
              f"restricted={r_survival:.12f}")
    print("A has cause counts (2,1), so its probability is capped at 1/2 and its total is split equally.")
    print("B has counts (1,3), so it stays unchanged. Shared a and block totals are unchanged.")
    print("The restriction changes cause probabilities, not the fitted minimum-lifetime distribution.")
    print("The order is provisional; these are hand-constructed teaching data.")


if __name__ == '__main__':
    main()
