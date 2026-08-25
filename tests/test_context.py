from dataclasses import FrozenInstanceError

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxctx.context import (
    ApplyReturn,
    InitReturn,
    get_parameter,
    wrap_random,
    set_parameter,
    transform,
    ScopedDict,
    scope,
)
from jaxctx.pytree import PureDataclassPytree


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


def test_return_containers_preserve_fields_tuple_protocol_and_pytree_roundtrip():
    collections = {'params': ScopedDict({'x': jnp.asarray(2.)})}
    apply_return = ApplyReturn(fn_val=jnp.asarray(1.), collections=collections)
    init_return = InitReturn(collections=collections)

    assert isinstance(apply_return, PureDataclassPytree)
    assert isinstance(init_return, PureDataclassPytree)
    assert not hasattr(apply_return, '__dict__')
    assert not hasattr(init_return, '__dict__')
    with pytest.raises(FrozenInstanceError):
        apply_return.fn_val = jnp.asarray(3.)
    with pytest.raises(FrozenInstanceError):
        init_return.collections = {}
    assert apply_return.fn_val == 1.
    assert apply_return.collections is collections
    assert len(apply_return) == 2
    assert apply_return[0] == apply_return.fn_val
    assert apply_return[1] is collections
    fn_val, unpacked_apply_collections = apply_return
    assert fn_val == apply_return.fn_val
    assert unpacked_apply_collections is collections

    assert init_return.collections is collections
    assert len(init_return) == 1
    assert init_return[0] is collections
    unpacked_init_collections, = init_return
    assert unpacked_init_collections is collections

    apply_leaves, apply_tree = jax.tree.flatten(apply_return)
    rebuilt_apply = jax.tree.unflatten(apply_tree, apply_leaves)
    assert isinstance(rebuilt_apply, ApplyReturn)
    assert rebuilt_apply.fn_val == apply_return.fn_val
    np.testing.assert_array_equal(
        rebuilt_apply.collections['params']['x'],
        apply_return.collections['params']['x']
    )

    init_leaves, init_tree = jax.tree.flatten(init_return)
    rebuilt_init = jax.tree.unflatten(init_tree, init_leaves)
    assert isinstance(rebuilt_init, InitReturn)
    np.testing.assert_array_equal(
        rebuilt_init.collections['params']['x'],
        init_return.collections['params']['x']
    )


def test_transformed_fn_is_slotted_and_frozen():
    transformed = transform(lambda: None)

    assert not hasattr(transformed, '__dict__')
    with pytest.raises(FrozenInstanceError):
        transformed.base_dtype = jnp.float16


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
