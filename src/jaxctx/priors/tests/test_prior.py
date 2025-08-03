import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax._src.scipy.special import logit

from jaxctx import transform
from jaxctx.priors.prior import quick_unit, quick_unit_inverse, Prior, tfpd, distribution_chain


def test_quick_unit():
    x = jnp.linspace(-10, 10, 1000000)
    y = quick_unit(x)
    assert np.all(y <= 1)
    assert np.all(y >= 0)
    x_reconstructed = quick_unit_inverse(y)
    np.testing.assert_allclose(x, x_reconstructed, atol=2e-5)

    g = jax.grad(quick_unit)
    assert np.all(np.isfinite(jax.vmap(g)(x)))
    assert np.isfinite(g(0.))

    h = jax.grad(quick_unit_inverse)
    assert np.all(np.isfinite(jax.vmap(h)(y)))
    assert np.isfinite(h(0.5))

    # Test performance against sigmoid and logit
    import time
    for f in [quick_unit, jax.nn.sigmoid]:
        g = jax.jit(f).lower(x).compile()
        t0 = time.time()
        for _ in range(1000):
            g(x).block_until_ready()
        print(f"{f.__name__} {time.time() - t0}s")

    for f in [quick_unit_inverse, logit]:
        g = jax.jit(f).lower(y).compile()
        t0 = time.time()
        for _ in range(1000):
            g(y).block_until_ready()
        print(f"{f.__name__} {time.time() - t0}s")


def test_prior():
    def model():
        x = Prior(tfpd.Normal(loc=0., scale=1.), name='x').sample()
        y = Prior(tfpd.Uniform(low=0., high=1.), name='y').parameter()
        z = Prior(tfpd.Beta(concentration0=0.5, concentration1=1.), name='z').realise()
        return x, y, z

    transformed_model = transform(model)
    params = transformed_model.init({'params': jax.random.PRNGKey(0)}, {}).collections
    print(params)

    print(transformed_model.apply({'params': jax.random.PRNGKey(0)}, params))


def test_no_quantile_prior():
    def prior_model():
        z = Prior(tfpd.VonMises(loc=0., concentration=1.)).sample()
        return z

    with pytest.raises(ValueError):
        transform(prior_model).init({'params': jax.random.PRNGKey(0)}, {})


def test_distribution_chain():
    d = tfpd.MultivariateNormalTriL(loc=jnp.zeros(5), scale_tril=jnp.eye(5))
    chain = distribution_chain(d)
    assert len(chain) == 2
    assert isinstance(chain[0], tfpd.Sample)
    assert isinstance(chain[1], tfpd.MultivariateNormalTriL)

    chain = distribution_chain(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    assert len(chain) == 1
    assert isinstance(chain[0], tfpd.Normal)


def test_priors():
    d = Prior(tfpd.Uniform(low=jnp.zeros(5), high=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Laplace(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Cauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.StudentT(df=1.5, loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfNormal(scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.HalfCauchy(loc=jnp.zeros(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gamma(concentration=jnp.ones(5), rate=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

    d = Prior(tfpd.Gumbel(loc=jnp.ones(5), scale=jnp.ones(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == ()
    assert d.shape == (5,)

    d = Prior(tfpd.MultivariateNormalTriL(loc=jnp.zeros(5), scale_tril=jnp.eye(5)))
    print(d)
    assert d.forward(jnp.ones(d.base_shape, jnp.float32)).shape == d.shape
    assert d.forward(jnp.zeros(d.base_shape, jnp.float32)).shape == d.shape
    assert d.base_shape == (5,)
    assert d.shape == (5,)

def test_various_collections():
    # We want to be able to create a model with
    def model():
        x = Prior(tfpd.Normal(loc=0., scale=1.), name='x').parameter()
        y = Prior(tfpd.Uniform(low=0., high=1.), name='y').realise()
        return x, y

    transformed_model = transform(model)
    params = transformed_model.init({'params': jax.random.PRNGKey(0), 'U': jax.random.PRNGKey(1)}, None).collections
    print(params)

    response = transformed_model.apply({'params': jax.random.PRNGKey(2), 'U': jax.random.PRNGKey(3)}, params)
    print(response)

