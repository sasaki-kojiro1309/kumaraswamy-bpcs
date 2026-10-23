"""Fit the existing teaching data; no uncertainty, Bayesian method, or simulation.

Run from the repository root: python -m experiments.stage4_example.
"""

from experiments.stage2_example import load_example
from src.mle import fit_direct, fit_mle, score


def main():
    data = load_example()
    profile, direct = fit_mle(data), fit_direct(data)
    for result in (profile, direct):
        print(f"{result.method}: status={result.status}, converged={result.converged}")
        print(f"a={result.a:.12f}; full log-likelihood={result.loglikelihood:.12f}")
        for label, pair in result.b_by_block.items():
            print(f"  {label}: b1={pair[0]:.12f}, b2={pair[1]:.12f}, "
                  f"P(cause 1)={pair[0]/sum(pair):.12f}")
        print(f"  max absolute log-parameter score={result.diagnostics['log_parameter_score_norm']:.3e}\n")
    print(f"Absolute profile/direct log-likelihood difference: {abs(profile.loglikelihood-direct.loglikelihood):.3e}")
    print("Raw score at the profile estimate:", score(data, profile.a, profile.b_by_block))
    print("\nThese are fits to hand-constructed teaching data, not a simulation study.")


if __name__ == "__main__":
    main()
