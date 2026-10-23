"""Reserved integration point; baseline parameters have no order restriction."""


def fit(data, *, restriction, **options):
    """Reserved interface; restriction and shared contracts are not finalized."""
    raise NotImplementedError("Order-restricted estimation is deferred")
