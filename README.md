# jaxctx

JAX Context for memoizationg collections of parameters, and handling sequences of random keys.
Based loosely on things like haiku, and flax, but not deep learning specific. No support for lifting things like
scan, etc. You must build these things on top of this if you want them.

Additional support added for probabilistic parameterisations based on priors.

## Packed constrained parameters

New models can combine all prior-constrained optimisation variables into one unconstrained parameter leaf. The pack
uses one base dtype and applies the real-to-unit transform once to the complete vector before passing static slices to
the individual priors.

```python
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxctx import transform
from jaxctx.priors import ParameterPack, Prior

tfpd = tfp.distributions

parameters = ParameterPack(
    [
        Prior(tfpd.Normal(0., 1.), name="location"),
        Prior(tfpd.Uniform(jnp.zeros(2), jnp.ones(2)), name="weights"),
    ],
    name="model",
)


def model():
    location, weights = parameters.parameter()
    return location, weights


transformed_model = transform(model, base_dtype=jnp.float32)
```

The trainable collection contains only `params["model"]`, with shape `(3,)`. The derived unit collection contains
`U["model"]`, while physical values and log probabilities remain available under their prior names. The pack is the
unit of optimisation and freezing; use separate packs when separate optimiser treatment is required.

`base_dtype` is configured once on `transform` and is shared by every unconstrained N value and unit-hypercube U value
in that transformed model, whether packed or created through `Prior.parameter()`. Priors remain responsible for their
physical X dtype, so a forward transform may produce floating, integer, boolean, or complex values independently of the
N/U dtype. Prior constructors therefore do not take `base_dtype`.

Packed parameterisation is intended for new models. Its checkpoint layout follows the declared prior order, so keep
the order and base dimensions stable when restoring a packed model. Existing models using `Prior.parameter()` remain
supported.

Physical initial values can be supplied in prior order. Entries may use the same constant or callable forms supported
by `Prior.parameter()`; callables receive the physical X shape and dtype of their prior:

```python
location, weights = parameters.parameter(
    init=(0., lambda key, shape, dtype: 0.5 * jnp.ones(shape, dtype))
)
```

## Periodic prior topology

A realised continuous prior can declare that every coordinate in its homogeneous base-space shape has equivalent
endpoints:

```python
import jax
import tensorflow_probability.substrates.jax as tfp

from jaxctx import transform
from jaxctx.priors import Prior

tfpd = tfp.distributions
phase = Prior(tfpd.Uniform(0., 1.), name="phase")


def model():
    return phase.realise(periodic=True)


collections, meta = transform(model).init(
    {"U": jax.random.PRNGKey(0)},
    {},
)
```

`meta.periodic` contains immutable `PeriodicEntry` records aligned by U collection, scope, prior name, and base shape.
The declaration is static topology metadata: it does not wrap U, change the prior measure, or alter X and log-probability
calculations. Canonical U values remain in `[0, 1)`.

`TransformMeta` is a zero-leaf static `Pytree`. Use its inherited `save`/`load` helpers for pickle persistence, or
`to_json`/`from_json` for the versioned JSON-safe representation.

`periodic=True` applies to the complete `base_shape` of one realised prior. Represent mixed periodic and non-periodic
variables as separate priors; partial periodicity within one prior and cyclic categorical variables are unsupported.
Changing topology requires constructing a new transformed model or explicitly retracing a static declaration.

The `InitReturn` tuple protocol has two values as of 1.2.0: `collections, meta`. Named access through
`result.collections` and `result.meta` is also supported. `ApplyReturn` is unchanged and does not carry metadata.

# Change Log

28 Aug, 2026 -- 1.2.0 adds init-only periodic base-space topology metadata for realised priors. `InitReturn` now unpacks
as `(collections, meta)`; `ApplyReturn` is unchanged.

25 Feb, 2026 -- 1.1.0 released with scoped dicts structure changes. Breaks backward compatibility with 1.0.x, but adds
support for nested contexts and more flexible scoping.

21 July, 2025 -- 1.0.3 released with support for `jaxctx.prior` and `jaxctx.prior.Prior`.

3 June, 2025 -- 1.0.2 prior constrained parameters released.

2 June, 2025 -- 1.0.1 released with context API.
