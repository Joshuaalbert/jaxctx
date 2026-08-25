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

# Change Log

25 Feb, 2026 -- 1.1.0 released with scoped dicts structure changes. Breaks backward compatibility with 1.0.x, but adds
support for nested contexts and more flexible scoping.

21 July, 2025 -- 1.0.3 released with support for `jaxctx.prior` and `jaxctx.prior.Prior`.

3 June, 2025 -- 1.0.2 prior constrained parameters released.

2 June, 2025 -- 1.0.1 released with context API.
