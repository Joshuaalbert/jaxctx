import warnings

import jax
import numpy as np
import pytest
from jax import numpy as jnp, vmap

from jaxctx.priors.prior import Prior
from jaxctx.priors.special_priors import get_interp_indices_and_weights, apply_interp, left_broadcast_multiply, \
    InterpolatedArray, Bernoulli, Categorical, Poisson, Beta, ForcedIdentifiability, UnnormalisedDirichlet, \
    _poisson_quantile_bisection, _poisson_quantile, Empirical, tfpd, TruncationWrapper, ExplicitDensityPrior


def test_get_interp_indices_and_weights():
    xp = [0, 1, 2, 3]
    x = 1.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 1
    assert alpha0 == 0.5
    assert i1 == 2
    assert alpha1 == 0.5

    x = 0
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0

    x = 3
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 2
    assert alpha0 == 0
    assert i1 == 3
    assert alpha1 == 1

    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == -1

    x = 4
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 4

    x = 5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert xp[i0] * alpha0 + xp[i1] * alpha1 == 5

    xp = [0., 0.]
    x = 0.
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1
    assert i1 == 1
    assert alpha1 == 0.

    xp = [0., 0.]
    x = -1
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 2.
    assert i1 == 1
    assert alpha1 == -1.

    xp = [0.]
    x = 0.5
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    assert i0 == 0
    assert alpha0 == 1.
    assert i1 == 0
    assert alpha1 == 0.

    # Vector ops
    xp = [0, 1, 2, 3]
    x = [1.5, 1.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(jnp.asarray(x, jnp.float32), jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 1]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 2]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    xp = [0, 1, 2, 3]
    x = [1.5, 2.5]
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(jnp.asarray(x, jnp.float32), jnp.asarray(xp, jnp.float32))
    print(i0, alpha0, i1, alpha1)
    np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))

    # xp = [0, 1, 2, 3]
    # x = [-0.5, 3.5]
    # (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, jnp.asarray(xp, jnp.float32))
    # print(i0, alpha0, i1, alpha1)
    # np.testing.assert_array_equal(i0, jnp.asarray([1, 2]))
    # np.testing.assert_array_equal(alpha0, jnp.asarray([0.5, 0.5]))
    # np.testing.assert_array_equal(i1, jnp.asarray([2, 3]))
    # np.testing.assert_array_equal(alpha1, jnp.asarray([0.5, 0.5]))


@pytest.mark.parametrize('regular_grid', [True, False])
def test_apply_interp(regular_grid):
    xp = jnp.linspace(0., 1., 10)
    x = jnp.linspace(-0.1, 1.1, 10)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    np.testing.assert_allclose(apply_interp(xp, i0, alpha0, i1, alpha1), x, atol=1e-6)

    x = jnp.linspace(-0.1, 1.1, 10)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 10, 6)

    x = 0.
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=regular_grid)
    assert apply_interp(jnp.zeros((4, 5, 10, 6)), i0, alpha0, i1, alpha1, axis=2).shape == (4, 5, 6)

    print(
        jax.jit(lambda x: apply_interp(x, i0, alpha0, i1, alpha1, axis=2)).lower(
            jnp.zeros((4, 5, 10, 6))).compile().cost_analysis()
    )
    # [{'bytes accessed': 1440.0, 'utilization1{}': 2.0, 'bytes accessed0{}': 960.0, 'bytes accessedout{}': 480.0, 'bytes accessed1{}': 960.0}]
    # [{'bytes accessed1{}': 960.0,  'utilization1{}': 2.0, 'bytes accessedout{}': 480.0, 'bytes accessed0{}': 960.0, 'bytes accessed': 1440.0}]


def test_regular_grid():
    # Inside bounds
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    x = jnp.linspace(0., 1., 100)
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )

    # Outside bounds
    x = jnp.linspace(-0.1, 1.1, 100)
    xp = jnp.linspace(0., 1., 10)
    fp = jax.random.normal(jax.random.PRNGKey(0), (10, 15))
    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=False)
    f_no = apply_interp(fp, i0, alpha0, i1, alpha1)

    (i0, alpha0), (i1, alpha1) = get_interp_indices_and_weights(x, xp, regular_grid=True)
    f_yes = apply_interp(fp, i0, alpha0, i1, alpha1)
    np.testing.assert_allclose(
        f_yes, f_no,
        atol=1e-6
    )


def test_left_broadcast_multiply():
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2,))) == np.ones((2, 3)))
    assert np.all(left_broadcast_multiply(np.ones((2, 3)), np.ones((2, 3))) == np.ones((2, 3)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=2) == np.ones((1, 2, 3, 4, 5)))
    assert np.all(
        left_broadcast_multiply(np.ones((1, 2, 3, 4, 5)), np.ones((3, 4)), axis=-3) == np.ones((1, 2, 3, 4, 5)))


@pytest.mark.parametrize('regular_grid', [True, False])
def test_interpolated_array(regular_grid: bool):
    # scalar time
    times = jnp.linspace(0, 10, 100)
    values = jnp.sin(times)
    interp = InterpolatedArray(times, values, regular_grid=regular_grid)
    assert interp(5.).shape == ()
    np.testing.assert_allclose(interp(5.), jnp.sin(5), atol=2e-3)

    # vector time
    assert interp(jnp.array([5., 6.])).shape == (2,)
    np.testing.assert_allclose(interp(jnp.array([5., 6.])), jnp.sin(jnp.array([5., 6.])), atol=2e-3)

    # Now with axis = 1
    times = jnp.linspace(0, 10, 100)
    values = jnp.stack([jnp.sin(times), jnp.cos(times)], axis=0)  # [2, 100]
    interp = InterpolatedArray(times, values, axis=1, regular_grid=regular_grid)
    assert interp(5.).shape == (2,)
    np.testing.assert_allclose(interp(5.), jnp.array([jnp.sin(5), jnp.cos(5)]), atol=2e-3)

    # Vector
    assert interp(jnp.array([5., 6., 7.])).shape == (2, 3)
    np.testing.assert_allclose(interp(jnp.array([5., 6., 7.])),
                               jnp.stack([jnp.sin(jnp.array([5., 6., 7.])), jnp.cos(jnp.array([5., 6., 7.]))],
                                         axis=0),
                               atol=2e-3)


@pytest.fixture(scope='module')
def mock_special_priors():
    # prior, (min, max), shape
    return [
        (Bernoulli(probs=jnp.ones(5), name='x'), (0, 1), (5,)),
        (Categorical(parametrisation='gumbel_max', probs=jnp.ones(5), name='x'), (0, 4), ()),
        (Categorical(parametrisation='cdf', probs=jnp.ones(5), name='x'), (0, 4), ()),
        (Poisson(rate=jnp.ones(5), name='x'), (0, 100), (5,)),
        (Beta(concentration0=jnp.ones(5), concentration1=jnp.ones(5), name='x'), (0, 1), (5,)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_left=True, name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_right=True, name='x'), (0, 1), (10, 5)),
        (ForcedIdentifiability(n=10, low=jnp.zeros(5), high=jnp.ones(5), fix_left=True, fix_right=True, name='x'),
         (0, 1), (10, 5)),
        (UnnormalisedDirichlet(concentration=jnp.ones(5), name='x'), (0, jnp.inf), (5,)),
        (UnnormalisedDirichlet(concentration=jnp.ones((3, 5)), name='x'), (0, jnp.inf), (3, 5)),
    ]


def test_special_priors(mock_special_priors):
    for prior, (vmin, vmax), shape in mock_special_priors:
        print(f"Testing {prior.__class__}")

        x = prior.forward(jnp.ones(prior.base_shape, prior.base_dtype))
        assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)
        assert x.shape == shape
        assert x.shape == prior.shape
        x = prior.forward(jnp.zeros(prior.base_shape, prior.base_dtype))
        assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)
        assert x.shape == shape
        assert x.shape == prior.shape

        u_input = vmap(lambda key: jax.random.uniform(key, shape=prior.base_shape))(jax.random.split(jax.random.PRNGKey(42), 10))
        x = vmap(lambda u: prior.forward(u))(u_input)
        assert jnp.all(x >= vmin)
        assert jnp.all(x <= vmax)

        try:
            u = vmap(lambda x: prior.inverse(x))(x)
            assert u.shape[1:] == prior.base_shape

            if prior.dtype in [jnp.bool_, jnp.int32, jnp.int64]:
                continue
            assert jnp.allclose(u, u_input)
        except NotImplementedError:
            warnings.warn(f"Skipping inverse test for {prior.__class__}")
            pass


@pytest.mark.parametrize("rate, error", (
        [2.0, 1.],
        [10., 1.],
        [100., 1.],
        [1000., 1.],
        [10000., 1.]
)
                         )
def test_poisson_quantile_bisection(rate, error):
    U = jnp.linspace(0., 1. - np.spacing(1.), 1000)
    x, x_results = _poisson_quantile_bisection(U, rate, unroll=False)
    diff_last_two = jnp.abs(x_results[..., -1] - x_results[..., -2])

    # Make sure less than 1 apart
    assert jnp.all(diff_last_two <= error)


@pytest.mark.parametrize("rate", [2.0, 10., 100., 1000., 10000.])
def test_poisson_quantile(rate):
    U = jnp.linspace(0., 1. - np.spacing(1.), 10000)
    x = _poisson_quantile(U, rate)
    assert jnp.all(jnp.isfinite(x))


def test_forced_identifiability():
    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.11, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.1, 0.2, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    assert jnp.allclose(prior.forward(u), x)

    x = jnp.asarray([0., 0.11, 0.1, 1.])
    prior = ForcedIdentifiability(n=4, low=0., high=1., fix_left=True, fix_right=True, name='x')
    u = prior.inverse(x)
    print(u)
    print(prior.forward(u))
    assert jnp.allclose(prior.forward(u), x)


def test_empirical():
    samples = jax.random.normal(jax.random.PRNGKey(42), shape=(2000,), dtype=jnp.float32)
    prior = Empirical(samples=samples, resolution=100, name='x')
    assert prior._percentiles.shape == (101, 1)

    x = prior.forward(jnp.ones(prior.base_shape, prior.base_dtype))
    assert x.shape == ()
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    x = prior.forward(jnp.zeros(prior.base_shape, prior.base_dtype))
    assert x.shape == ()
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))

    x = prior.forward(0.5 * jnp.ones(prior.base_shape, prior.base_dtype))
    np.testing.assert_allclose(x, 0., atol=0.06)

    u_input = vmap(lambda key: jax.random.uniform(key, shape=prior.base_shape, dtype=prior.base_dtype))(
        jax.random.split(jax.random.PRNGKey(42), 1000))
    x = vmap(lambda u: prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))

    u = vmap(lambda x: prior.inverse(x))(x)
    # print(u)
    np.testing.assert_allclose(u, u_input, atol=5e-7)
    assert u.shape[1:] == prior.base_shape


def test_truncation_wrapper():
    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=0., high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: jax.random.uniform(key, shape=trancated_prior.base_shape))(
        jax.random.split(jax.random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=-jnp.inf, high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: jax.random.uniform(key, shape=trancated_prior.base_shape))(
        jax.random.split(jax.random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= -jnp.inf)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    prior = Prior(tfpd.Normal(loc=jnp.zeros(5), scale=0.01 * jnp.ones(5)))
    trancated_prior = TruncationWrapper(prior=prior, low=0., high=1.)

    x = trancated_prior.forward(jnp.ones(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    x = trancated_prior.forward(jnp.zeros(trancated_prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)
    assert x.shape == (5,)

    u_input = vmap(lambda key: jax.random.uniform(key, shape=trancated_prior.base_shape))(
        jax.random.split(jax.random.PRNGKey(42), 1000))
    x = vmap(lambda u: trancated_prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: trancated_prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)


def test_explicit_density_prior():
    resolution = 10
    density = jnp.ones((resolution + 1, resolution))
    axes = (jnp.linspace(0, 1, resolution + 1), jnp.linspace(0, 1, resolution))
    prior = ExplicitDensityPrior(axes=axes, density=density, regular_grid=True)

    x = prior.forward(jnp.ones(prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x == 1.)
    assert x.shape == (2,)

    x = prior.forward(jnp.zeros(prior.base_shape, prior.base_dtype))
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x == 0.)
    assert x.shape == (2,)

    u_input = vmap(lambda key: jax.random.uniform(key, shape=prior.base_shape))(jax.random.split(jax.random.PRNGKey(42), 1000))
    x = vmap(lambda u: prior.forward(u))(u_input)
    assert jnp.all(jnp.bitwise_not(jnp.isnan(x)))
    assert jnp.all(x >= 0.)
    assert jnp.all(x <= 1.)

    u = vmap(lambda x: prior.inverse(x))(x)
    np.testing.assert_allclose(u, u_input, atol=5e-7)

    assert jnp.all(jnp.isfinite(jax.vmap(prior.log_prob)(x)))
