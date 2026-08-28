import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxctx import PeriodicEntry, ScopedDict, TransformMeta, scope, transform
from jaxctx.priors.prior import Prior, realise_prior, tfpd
from jaxctx.priors.special_priors import Bernoulli


def _continuous_prior(name, shape=()):
    return Prior(
        tfpd.Uniform(
            low=jnp.zeros(shape, dtype=jnp.float32),
            high=jnp.ones(shape, dtype=jnp.float32),
        ),
        name=name,
    )


def test_periodic_metadata_records_whole_priors_in_canonical_order():
    phase = _continuous_prior('phase')
    calibration = _continuous_prior('calibration', (2,))
    auxiliary = _continuous_prior('auxiliary')

    def model():
        with scope('source'):
            phase.realise(periodic=True)
            calibration.realise()
        auxiliary.realise(periodic=True, U_collection='latent')

    result = transform(model).init(
        {'U': jax.random.PRNGKey(0)},
        {},
    )

    assert result.meta == TransformMeta(periodic=(
        PeriodicEntry('U', ('source',), 'calibration', (2,), False),
        PeriodicEntry('U', ('source',), 'phase', (), True),
        PeriodicEntry('latent', (), 'auxiliary', (), True),
    ))
    assert result.collections['U']['source']['calibration'].shape == (2,)
    assert result.collections['U']['source']['phase'].shape == ()
    assert result.collections['latent']['auxiliary'].shape == ()
    assert 'periodic' not in result.collections


def test_periodic_declaration_does_not_change_u_x_or_log_probability():
    prior = _continuous_prior('phase', (3,))

    def model(periodic):
        return prior.realise(periodic=periodic)

    key = jax.random.PRNGKey(1)
    nonperiodic = transform(lambda: model(False)).init({'U': key}, {})
    periodic = transform(lambda: model(True)).init({'U': key}, {})

    for collection in ('U', 'X', 'log_prob'):
        np.testing.assert_array_equal(
            nonperiodic.collections[collection]['phase'],
            periodic.collections[collection]['phase'],
        )
    assert nonperiodic.meta.periodic[0].periodic is False
    assert periodic.meta.periodic[0].periodic is True


def test_duplicate_periodic_declaration_is_idempotent():
    prior = _continuous_prior('phase')

    def model():
        first = prior.realise(periodic=True)
        second = prior.realise(periodic=True)
        return first, second

    result = transform(model).init({'U': jax.random.PRNGKey(2)}, {})

    assert len(result.meta.periodic) == 1
    assert result.meta.periodic[0].periodic is True


def test_conflicting_periodic_declarations_fail_during_init():
    prior = _continuous_prior('phase')

    def model():
        prior.realise(periodic=True)
        prior.realise(periodic=False)

    with pytest.raises(ValueError, match='Conflicting periodic declarations'):
        transform(model).init({'U': jax.random.PRNGKey(3)}, {})


@pytest.mark.parametrize(
    'periodic',
    [
        1,
        np.bool_(True),
        [True],
        np.asarray(True),
        jnp.asarray(True),
    ],
)
def test_periodic_declaration_accepts_only_python_bool(periodic):
    prior = _continuous_prior('phase')

    def model():
        return prior.realise(periodic=periodic)

    with pytest.raises(TypeError, match='must be a Python bool'):
        transform(model).init({'U': jax.random.PRNGKey(4)}, {})


def test_traced_periodic_declaration_fails_during_tracing():
    prior = _continuous_prior('phase')

    def model(periodic):
        return prior.realise(periodic=periodic)

    transformed = transform(model)
    init = jax.jit(transformed.init)

    with pytest.raises(TypeError, match='must be a Python bool'):
        init({'U': jax.random.PRNGKey(5)}, {}, jnp.asarray(True))


def test_python_bool_can_be_an_explicit_static_jit_specialisation():
    prior = _continuous_prior('phase')

    def model(periodic):
        return prior.realise(periodic=periodic)

    init = jax.jit(transform(model).init, static_argnums=2)

    nonperiodic = init({'U': jax.random.PRNGKey(11)}, {}, False)
    periodic = init({'U': jax.random.PRNGKey(11)}, {}, True)

    assert nonperiodic.meta.periodic == (PeriodicEntry('U', (), 'phase', (), False),)
    assert periodic.meta.periodic == (PeriodicEntry('U', (), 'phase', (), True),)


def test_discrete_prior_cannot_be_declared_periodic():
    prior = Bernoulli(probs=0.5, name='choice')

    def model():
        return prior.realise(periodic=True)

    with pytest.raises(TypeError, match='Cyclic discrete variables are unsupported'):
        transform(model).init({'U': jax.random.PRNGKey(6)}, {})


def test_realise_prior_function_exposes_periodic_declaration():
    prior = _continuous_prior('phase')

    def model():
        return realise_prior(prior, periodic=True)

    result = transform(model).init({'U': jax.random.PRNGKey(7)}, {})

    assert result.meta.periodic == (PeriodicEntry('U', (), 'phase', (), True),)


def test_realise_prior_preserves_legacy_positional_collection_arguments():
    prior = _continuous_prior('phase')

    def model():
        return realise_prior(prior, 'latent', 'physical', 'density', 'U', periodic=True)

    result = transform(model).init({'U': jax.random.PRNGKey(10)}, {})

    assert result.meta.periodic == (PeriodicEntry('latent', (), 'phase', (), True),)
    assert 'phase' in result.collections['latent']
    assert 'phase' in result.collections['physical']
    assert 'phase' in result.collections['density']


def test_apply_return_and_hlo_do_not_contain_periodic_metadata():
    prior = _continuous_prior('phase', (4,))

    def make_transformed(periodic):
        def model():
            return prior.realise(periodic=periodic)

        return transform(model)

    nonperiodic = make_transformed(False)
    periodic = make_transformed(True)
    key = jax.random.PRNGKey(8)
    nonperiodic_collections = nonperiodic.init({'U': key}, {}).collections
    periodic_collections = periodic.init({'U': key}, {}).collections

    def apply_x(transformed, collections):
        return transformed.apply({}, collections).fn_val

    nonperiodic_apply = jax.jit(lambda collections: apply_x(nonperiodic, collections))
    periodic_apply = jax.jit(lambda collections: apply_x(periodic, collections))
    nonperiodic_hlo = nonperiodic_apply.lower(nonperiodic_collections).compiler_ir('hlo').as_hlo_text()
    periodic_hlo = periodic_apply.lower(periodic_collections).compiler_ir('hlo').as_hlo_text()

    np.testing.assert_array_equal(
        nonperiodic_apply(nonperiodic_collections),
        periodic_apply(periodic_collections),
    )
    assert nonperiodic_hlo == periodic_hlo
    apply_result = periodic.apply({}, periodic_collections)
    assert not hasattr(apply_result, 'meta')
    assert all(not isinstance(leaf, TransformMeta) for leaf in jax.tree.leaves(apply_result))


def test_jitted_init_returns_metadata_as_static_zero_leaf_pytree():
    prior = _continuous_prior('phase')

    def model():
        return prior.realise(periodic=True)

    result = jax.jit(transform(model).init)({'U': jax.random.PRNGKey(9)}, {})

    assert result.meta.periodic == (PeriodicEntry('U', (), 'phase', (), True),)
    assert result.meta not in jax.tree.leaves(result)
    assert all(hasattr(leaf, 'dtype') for leaf in jax.tree.leaves(result))


def test_empty_model_has_empty_transform_metadata():
    result = transform(lambda: None).init({}, {'state': ScopedDict()})

    assert result.meta == TransformMeta()
