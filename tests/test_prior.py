import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.scipy import special as jsp

from jaxctx import transform
from jaxctx.priors.prior import AbstractPrior, quick_unit, quick_unit_inverse, Prior, tfpd, distribution_chain


class SpyPrior(AbstractPrior):
    def __init__(self, state, name="p"):
        super().__init__(name=name, base_dtype=jnp.float32)
        self._state = state

    def _dtype(self):
        return jnp.float32

    def _base_shape(self):
        return ()

    def _shape(self):
        return ()

    def _forward(self, U):
        return U

    def _inverse(self, X):
        self._state["inverse_calls"] += 1
        if self._state["mode"] == "bad":
            raise RuntimeError("inverse() was called during apply")
        return X

    def _log_prob(self, X):
        return jnp.asarray(0.0, dtype=jnp.float32)


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

    for f in [quick_unit_inverse, jsp.logit]:
        g = jax.jit(f).lower(y).compile()
        t0 = time.time()
        for _ in range(1000):
            g(y).block_until_ready()
        print(f"{f.__name__} {time.time() - t0}s")


def _tail_threshold(forward, inverse, dtype, max_x: float = 1e4, iters: int = 80) -> float:
    def _is_valid(x_value: float) -> bool:
        x_arr = jnp.asarray(x_value, dtype=dtype)
        y = forward(x_arr)
        x_rec = inverse(y)
        return bool((y > 0) & (y < 1) & jnp.isfinite(y) & jnp.isfinite(x_rec))

    lo = 0.0
    hi = 1.0
    while hi < max_x and _is_valid(hi):
        lo = hi
        hi *= 2.0

    if _is_valid(hi):
        return float(hi)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _is_valid(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def test_quick_unit_tail_saturation():
    jax.config.update("jax_enable_x64", True)
    assert jax.config.read("jax_enable_x64")

    thresholds = {}
    for dtype in (jnp.float32, jnp.float64):
        thresholds[(dtype, "sigmoid")] = _tail_threshold(
            jax.nn.sigmoid,
            jsp.logit,
            dtype
        )
        thresholds[(dtype, "ndtr")] = _tail_threshold(
            jsp.ndtr,
            jsp.ndtri,
            dtype
        )
        print(f"{dtype} sigmoid tail threshold: {thresholds[(dtype, 'sigmoid')]}")
        print(f"{dtype} ndtr tail threshold: {thresholds[(dtype, 'ndtr')]}")

        assert thresholds[(dtype, "sigmoid")] > 0
        assert thresholds[(dtype, "ndtr")] > 0
        assert thresholds[(dtype, "sigmoid")] >= thresholds[(dtype, "ndtr")]

    assert thresholds[(jnp.float64, "sigmoid")] >= thresholds[(jnp.float32, "sigmoid")]
    assert thresholds[(jnp.float64, "ndtr")] >= thresholds[(jnp.float32, "ndtr")]


def test_prior():
    def model():
        x = Prior(tfpd.Normal(loc=0., scale=1.), name='x').realise()
        y = Prior(tfpd.Uniform(low=0., high=1.), name='y').parameter()
        y_rand = Prior(tfpd.Uniform(low=0., high=1.), name='y_rand').parameter(random_init=True)
        y_init = Prior(tfpd.Uniform(low=0., high=1.), name='y_init').parameter(init=0.75)
        y_init_callable = Prior(tfpd.Uniform(low=0., high=1.), name='y_init_callable').parameter(
            init=lambda key, shape, dtype: 0.75 * jnp.ones(shape, dtype))
        z = Prior(tfpd.Beta(concentration0=0.5, concentration1=1.), name='z').realise()
        return x, y, y_rand, y_init, y_init_callable, z

    transformed_model = transform(model)
    params = transformed_model.init({'params': jax.random.PRNGKey(0), 'U': jax.random.PRNGKey(1)}, {}).collections
    print(params)
    y_init = params['X']['y_init']
    assert np.isclose(y_init, 0.75, atol=1e-5)
    y_init_callable = params['X']['y_init_callable']
    assert np.isclose(y_init_callable, 0.75, atol=1e-5)

    print(transformed_model.apply({}, params))


def test_no_quantile_prior():
    def prior_model():
        z = Prior(tfpd.VonMises(loc=0., concentration=1.)).realise()
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


def test_prior_parameter_const_init_does_not_touch_inverse_on_apply():
    state = {"mode": "safe", "inverse_calls": 0}
    prior = SpyPrior(state, name="p")

    def f(x):
        p = prior.parameter(init=jnp.asarray(0.5, dtype=jnp.float32))
        return x + p

    tf = transform(f)
    init_res = tf.init({}, {}, jnp.asarray(1.0, dtype=jnp.float32))

    assert state["inverse_calls"] == 1

    state["mode"] = "bad"
    out = tf.apply({}, {"params": init_res.collections["params"]}, jnp.asarray(2.0, dtype=jnp.float32)).fn_val

    assert out == jnp.asarray(2.5, dtype=jnp.float32)
    assert state["inverse_calls"] == 1


def test_prior_parameter_traced_const_init_does_not_touch_inverse_on_jitted_apply():
    state = {"mode": "safe", "inverse_calls": 0}
    prior = SpyPrior(state, name="p")

    def f(x):
        init_value = jnp.sin(x) + jnp.asarray(0.5, dtype=jnp.float32)
        p = prior.parameter(init=init_value)
        return x + p

    tf = transform(f)
    init_res = tf.init({}, {}, jnp.asarray(1.0, dtype=jnp.float32))

    assert state["inverse_calls"] == 1

    state["mode"] = "bad"
    apply_fn = jax.jit(lambda x, p: tf.apply({}, {"params": p}, x).fn_val)
    out = apply_fn(jnp.asarray(2.0, dtype=jnp.float32), init_res.collections["params"])

    assert out.shape == ()
    assert state["inverse_calls"] == 1


def test_prior_parameter_callable_init_does_not_touch_inverse_on_apply():
    state = {"mode": "safe", "inverse_calls": 0}
    prior = SpyPrior(state, name="p")

    def f(x):
        def init_fn(key, shape, dtype):
            del key, shape
            return jnp.asarray(0.5, dtype=dtype)

        p = prior.parameter(init=init_fn)
        return x + p

    tf = transform(f)
    init_res = tf.init({"params": jax.random.PRNGKey(0)}, {}, jnp.asarray(1.0, dtype=jnp.float32))

    assert state["inverse_calls"] == 1

    state["mode"] = "bad"
    out = tf.apply({}, {"params": init_res.collections["params"]}, jnp.asarray(2.0, dtype=jnp.float32)).fn_val

    assert out == jnp.asarray(2.5, dtype=jnp.float32)
    assert state["inverse_calls"] == 1


def test_prior_parameter_split_jit_keeps_sin_in_init_hlo_only():
    state = {"mode": "safe", "inverse_calls": 0}
    prior = SpyPrior(state, name="p")

    def f(x):
        p = prior.parameter(init=jnp.sin(x))
        return x + p

    @jax.jit
    def init_fn(x):
        tf = transform(f)
        return tf.init({}, {}, x).collections["params"]

    @jax.jit
    def apply_fn(x, params):
        tf = transform(f)
        return tf.apply({}, {"params": params}, x).fn_val

    x = jnp.asarray(1.0, dtype=jnp.float32)
    params = init_fn(x)

    init_hlo = init_fn.lower(x).compiler_ir(dialect="hlo").as_hlo_text().lower()
    apply_hlo = apply_fn.lower(x, params).compiler_ir(dialect="hlo").as_hlo_text().lower()

    assert "sine" in init_hlo or "sin" in init_hlo
    assert "sine" not in apply_hlo and "sin" not in apply_hlo

    inverse_calls_before_apply = state["inverse_calls"]

    state["mode"] = "bad"
    out = apply_fn(x, params)

    assert out.shape == ()
    assert state["inverse_calls"] == inverse_calls_before_apply
