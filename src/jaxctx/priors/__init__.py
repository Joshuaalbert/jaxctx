__all__ = [
    'AbstractPrior',
    'ParameterPack',
    'Prior',
]


def __getattr__(name):
    # context imports priors.types while jaxctx itself is initialising, so
    # public prior imports must remain lazy.
    if name in __all__:
        from jaxctx.priors import prior
        return getattr(prior, name)
    raise AttributeError(name)
