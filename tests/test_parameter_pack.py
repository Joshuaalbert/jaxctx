import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxctx import ScopedDict, scope, transform
from jaxctx.priors import ParameterPack
from jaxctx.priors.prior import AbstractPrior, Prior, quick_unit, quick_unit_inverse, tfpd


class ExpandingPrior(AbstractPrior):
    """Two unit coordinates mapped to three physical coordinates."""

    def __init__(self, name="expanded"):
        super().__init__(name=name, base_dtype=jnp.float32)

    def _dtype(self):
        return jnp.float32

    def _base_shape(self):
        return (2,)

    def _shape(self):
        return (3,)

    def _forward(self, U):
        return jnp.concatenate([U, jnp.sum(U, keepdims=True)])

    def _inverse(self, X):
        return X[:2]

    def _log_prob(self, X):
        del X
        return jnp.asarray(0., dtype=jnp.float32)


def _uniform(name, shape=(), base_dtype=jnp.float32):
    return Prior(
        tfpd.Uniform(low=jnp.zeros(shape), high=jnp.ones(shape)),
        name=name,
        base_dtype=base_dtype
    )


def test_parameter_pack_uses_one_packed_parameter_and_unit_value():
    pack = ParameterPack([_uniform("x"), _uniform("y", (2,))], name="theta")

    def model():
        return pack.parameter()

    result = transform(model).init({}, {})

    assert tuple(result.collections["params"].keys()) == ("theta",)
    assert result.collections["params"]["theta"].shape == (3,)
    assert result.collections["params"]["theta"].dtype == pack.base_dtype
    np.testing.assert_allclose(result.collections["params"]["theta"], jnp.zeros(3))

    assert tuple(result.collections["U"].keys()) == ("theta",)
    np.testing.assert_allclose(result.collections["U"]["theta"], 0.5 * jnp.ones(3))
    np.testing.assert_allclose(result.collections["X"]["x"], 0.5)
    np.testing.assert_allclose(result.collections["X"]["y"], 0.5 * jnp.ones(2))
    assert set(result.collections["log_prob"].keys()) == {"x", "y"}


def test_parameter_pack_supports_explicit_and_callable_initial_values():
    pack = ParameterPack([_uniform("x"), _uniform("y", (2,))], name="theta")

    def y_initialiser(key, shape, dtype):
        del key
        return 0.75 * jnp.ones(shape, dtype)

    def model():
        return pack.parameter(init=(0.25, y_initialiser))

    key = jax.random.PRNGKey(1)
    result = transform(model).init({"params": key}, {})
    expected_U = jnp.asarray([0.25, 0.75, 0.75], dtype=pack.base_dtype)

    np.testing.assert_allclose(result.collections["params"]["theta"], quick_unit_inverse(expected_U), rtol=1e-6)
    np.testing.assert_allclose(result.collections["U"]["theta"], expected_U, rtol=1e-6)
    np.testing.assert_allclose(result.collections["X"]["x"], expected_U[0], rtol=1e-6)
    np.testing.assert_allclose(result.collections["X"]["y"], expected_U[1:], rtol=1e-6)


def test_parameter_pack_random_initialises_the_packed_leaf_in_one_draw():
    pack = ParameterPack([_uniform("x"), _uniform("y", (2,))], name="theta")

    def model():
        return pack.parameter(random_init=True)

    key = jax.random.PRNGKey(7)
    result = transform(model).init({"params": key}, {})
    _, draw_key = jax.random.split(key)
    expected = jax.random.normal(draw_key, (3,), pack.base_dtype)

    np.testing.assert_array_equal(result.collections["params"]["theta"], expected)


def test_parameter_pack_supports_different_base_and_physical_shapes():
    pack = ParameterPack([_uniform("x"), ExpandingPrior()], name="theta")

    def model():
        return pack.parameter()

    result = transform(model).init({}, {})

    assert result.collections["params"]["theta"].shape == (3,)
    assert result.collections["X"]["x"].shape == ()
    assert result.collections["X"]["expanded"].shape == (3,)
    np.testing.assert_allclose(result.collections["X"]["expanded"], jnp.asarray([0.5, 0.5, 1.]))


def test_parameter_pack_requires_one_base_dtype_and_unique_names():
    with pytest.raises(ValueError, match="at least one prior"):
        ParameterPack([])

    with pytest.raises(ValueError, match="names must be unique"):
        ParameterPack([_uniform("x"), _uniform("x")])

    with pytest.raises(ValueError, match="must have base dtype"):
        ParameterPack([_uniform("x", base_dtype=jnp.float32), _uniform("y", base_dtype=jnp.float16)])


def test_parameter_pack_validates_initial_value_count():
    pack = ParameterPack([_uniform("x"), _uniform("y")], name="theta")

    def model():
        return pack.parameter(init=(0.25,))

    with pytest.raises(ValueError, match="expected 2 initial values"):
        transform(model).init({}, {})


def test_parameter_pack_gradient_matches_direct_packed_transform():
    x_prior = _uniform("x")
    y_prior = _uniform("y", (2,))
    pack = ParameterPack([x_prior, y_prior], name="theta")

    def model():
        x, y = pack.parameter()
        return x + jnp.sum(jnp.square(y))

    transformed = transform(model)

    def actual_loss(N):
        params = ScopedDict({"theta": N})
        return transformed.apply({}, {"params": params}).fn_val

    def expected_loss(N):
        U = quick_unit(N)
        x = x_prior.forward(jax.lax.reshape(U[:1], ()))
        y = y_prior.forward(U[1:])
        return x + jnp.sum(jnp.square(y))

    N = jnp.asarray([-0.5, 0.25, 1.], dtype=pack.base_dtype)
    np.testing.assert_allclose(actual_loss(N), expected_loss(N), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(actual_loss)(N), jax.grad(expected_loss)(N), rtol=1e-6)


def test_derived_collections_are_refreshed_from_current_parameters():
    pack = ParameterPack([_uniform("x"), _uniform("y")], name="theta")

    def packed_model():
        return pack.parameter()

    packed_transform = transform(packed_model)
    packed_collections = packed_transform.init({}, {}).collections
    packed_collections["params"]["theta"] = jnp.ones(2, dtype=pack.base_dtype)
    packed_result = packed_transform.apply({}, packed_collections)
    expected_U = quick_unit(jnp.ones(2, dtype=pack.base_dtype))

    np.testing.assert_allclose(packed_result.collections["U"]["theta"], expected_U)
    np.testing.assert_allclose(packed_result.collections["X"]["x"], expected_U[0])
    np.testing.assert_allclose(packed_result.collections["X"]["y"], expected_U[1])

    prior = _uniform("legacy")

    def legacy_model():
        return prior.parameter()

    legacy_transform = transform(legacy_model)
    legacy_collections = legacy_transform.init({}, {}).collections
    legacy_collections["params"]["legacy"] = jnp.ones((), dtype=prior.base_dtype)
    legacy_result = legacy_transform.apply({}, legacy_collections)

    np.testing.assert_allclose(legacy_result.collections["U"]["legacy"], expected_U[0])
    np.testing.assert_allclose(legacy_result.collections["X"]["legacy"], expected_U[0])


def test_parameter_pack_preserves_scope_for_all_collections():
    pack = ParameterPack([_uniform("x"), _uniform("y")], name="theta")

    def model():
        with scope("layer"):
            return pack.parameter()

    collections = transform(model).init({}, {}).collections

    assert collections["params"]["layer"]["theta"].shape == (2,)
    assert collections["U"]["layer"]["theta"].shape == (2,)
    assert collections["X"]["layer"]["x"].shape == ()
    assert collections["log_prob"]["layer"]["y"].shape == ()


def test_parameter_pack_apply_graph_has_one_unit_transform_and_no_pack_concatenation():
    num_priors = 16
    priors = tuple(_uniform(f"p{i}") for i in range(num_priors))
    pack = ParameterPack(priors, name="theta")

    def packed_model():
        return sum(pack.parameter())

    def legacy_model():
        return sum(prior.parameter() for prior in priors)

    packed_transform = transform(packed_model)
    legacy_transform = transform(legacy_model)

    def packed_apply(N):
        return packed_transform.apply({}, {"params": ScopedDict({"theta": N})}).fn_val

    def legacy_apply(*values):
        params = ScopedDict({prior.name: value for prior, value in zip(priors, values)})
        return legacy_transform.apply({}, {"params": params}).fn_val

    packed_N = jnp.zeros((num_priors,), dtype=pack.base_dtype)
    legacy_N = (jnp.zeros((), dtype=pack.base_dtype),) * num_priors
    packed_hlo = jax.jit(packed_apply).lower(packed_N).compiler_ir(dialect="hlo").as_hlo_text().lower()
    legacy_hlo = jax.jit(legacy_apply).lower(*legacy_N).compiler_ir(dialect="hlo").as_hlo_text().lower()

    assert "concatenate" not in packed_hlo
    assert packed_hlo.count("exponential") < legacy_hlo.count("exponential")
