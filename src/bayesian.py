"""Reserved teammate integration point. Bayesian estimation is not implemented.

Future estimators should reuse model.py and the shared validated block data and
likelihood interfaces once those milestones exist. No prior is assumed here.
"""


def fit(data, *, prior, **options):
    """Reserved interface; final data/result contracts will follow the MLE stage."""
    raise NotImplementedError("Bayesian estimation is deferred to the teammate")
