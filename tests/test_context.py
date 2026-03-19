import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxctx.context import get_parameter, wrap_random, set_parameter, transform, ScopedDict, scope


def test_transform():
    with jax.checking_leaks():
        def f(x) -> jax.Array:
            y = get_parameter(
                'y', 'params', (), jnp.float32,
                init=wrap_random(jax.random.normal, 'params')
            )
            s = get_parameter('s', 'state', y.shape, y.dtype, init=jnp.zeros)
            s = set_parameter('s', 'state', s + x + y)
            return s

        transformed = transform(f)

        init = jax.jit(transformed.init)({'params': jax.random.PRNGKey(0)}, {}, 1)
        print(init)

        apply = jax.jit(transformed.apply)

        response = apply({'params': jax.random.PRNGKey(0)}, init.collections, 1)
        print(response)
        assert response.fn_val == 1 + response.collections['params']['y']
        assert response.fn_val == response.collections['state']['s']
        for key, val in response.collections.items():
            assert isinstance(val, ScopedDict)

        next_response = apply({'params': jax.random.PRNGKey(0)}, response.collections, 1)

        print(next_response)
        assert next_response.fn_val == response.collections['state']['s'] + 1 + response.collections['params']['y']
        assert next_response.fn_val == next_response.collections['state']['s']


def test_scoped_dict_nested_and_dotted():
    with jax.checking_leaks():
        def f(x) -> jax.Array:
            with scope('layer'):
                y = get_parameter(
                    'y', 'params', (), jnp.float32,
                    init=wrap_random(jax.random.normal, 'params')
                )
                s = get_parameter('s', 'state', y.shape, y.dtype, init=jnp.zeros)
                s = set_parameter('s', 'state', s + x + y)
            return s

        transformed = transform(f)

        init = jax.jit(transformed.init)({'params': jax.random.PRNGKey(0)}, {}, 1)
        apply = jax.jit(transformed.apply)

        response = apply({'params': jax.random.PRNGKey(0)}, init.collections, 1)
        assert response.fn_val == 1 + response.collections['params']['layer']['y']
        assert response.fn_val == response.collections['state']['layer']['s']

        params = response.collections['params']
        np.testing.assert_array_equal(
            params.get_dotted('.layer.y'),
            response.collections['params']['layer']['y']
        )

        items = dict(params.iter_items())
        assert 'layer.y' in items
        np.testing.assert_array_equal(
            items['layer.y'],
            response.collections['params']['layer']['y']
        )


def test_scoped_dict_set_dotted():
    scoped = ScopedDict()

    scoped.set_dotted('layer.weight', 1)
    scoped.set_dotted('.layer.bias', 2)

    assert scoped.to_dict() == {'layer': {'weight': 1, 'bias': 2}}
    assert scoped.get_dotted('layer.weight') == 1
    assert scoped.get_dotted('.layer.bias') == 2

    with pytest.raises(ValueError, match="Dotted key cannot be empty."):
        scoped.set_dotted('', 3)

    with pytest.raises(ValueError, match="Cannot overwrite scope 'layer' with a leaf value."):
        scoped.set_dotted('layer', 4)

    leaf_collision = ScopedDict({'layer': 1})
    with pytest.raises(ValueError, match="Scope 'layer' collides with existing leaf."):
        leaf_collision.set_dotted('layer.weight', 2)

    with pytest.raises(ValueError, match="Cannot overwrite leaf 'weight' with a scope."):
        scoped.set_dotted('layer.weight', {'value': 5})


def test_scoped_dict_iter_items_order_and_scopes():
    scoped = ScopedDict({
        'b': 2,
        'a': {
            'd': 4,
            'c': 3,
        },
        'z': {
            'alpha': 1,
        },
    })

    assert list(scoped.iter_items()) == [
        ('a.c', 3),
        ('a.d', 4),
        ('b', 2),
        ('z.alpha', 1),
    ]

    assert list(scoped.with_scopes(['a']).iter_items()) == [
        ('a.c', 3),
        ('a.d', 4),
    ]

    assert list(scoped.with_scopes(['missing']).iter_items()) == []
