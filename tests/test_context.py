import json
import pickle
from dataclasses import FrozenInstanceError

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxctx.context import (
    ApplyReturn,
    InitReturn,
    PeriodicEntry,
    ScopedDict,
    TransformMeta,
    get_parameter,
    scope,
    set_parameter,
    transform,
    wrap_random,
)
from jaxctx.pytree import PureDataclassPytree, Pytree


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
    meta = TransformMeta(periodic=(
        PeriodicEntry(
            collection='U',
            scope=('model',),
            name='phase',
            base_shape=(),
            periodic=True,
        ),
    ))
    apply_return = ApplyReturn(fn_val=jnp.asarray(1.), collections=collections)
    init_return = InitReturn(collections=collections, meta=meta)

    assert isinstance(apply_return, PureDataclassPytree)
    assert isinstance(init_return, PureDataclassPytree)
    assert not hasattr(apply_return, '__dict__')
    assert not hasattr(init_return, '__dict__')
    with pytest.raises(FrozenInstanceError):
        apply_return.fn_val = jnp.asarray(3.)
    with pytest.raises(FrozenInstanceError):
        init_return.collections = {}
    with pytest.raises(FrozenInstanceError):
        init_return.meta = TransformMeta()
    assert apply_return.fn_val == 1.
    assert apply_return.collections is collections
    assert len(apply_return) == 2
    assert apply_return[0] == apply_return.fn_val
    assert apply_return[1] is collections
    fn_val, unpacked_apply_collections = apply_return
    assert fn_val == apply_return.fn_val
    assert unpacked_apply_collections is collections

    assert init_return.collections is collections
    assert init_return.meta is meta
    assert len(init_return) == 2
    assert init_return[0] is collections
    assert init_return[1] is meta
    unpacked_init_collections, unpacked_meta = init_return
    assert unpacked_init_collections is collections
    assert unpacked_meta is meta

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
    assert rebuilt_init.meta == meta

    unpickled_init = pickle.loads(pickle.dumps(init_return))
    np.testing.assert_array_equal(
        unpickled_init.collections['params']['x'],
        init_return.collections['params']['x'],
    )
    assert unpickled_init.meta == meta


def test_transform_meta_is_static_hashable_persistable_and_json_safe(tmp_path):
    meta = TransformMeta(periodic=(
        PeriodicEntry(
            collection='U',
            scope=(),
            name='amplitude',
            base_shape=(2,),
            periodic=False,
        ),
        PeriodicEntry(
            collection='U',
            scope=('source',),
            name='phase',
            base_shape=(),
            periodic=True,
        ),
    ))

    assert isinstance(meta, Pytree)
    assert jax.tree.leaves(meta) == []
    assert not hasattr(meta, '__dict__')
    assert not hasattr(meta.periodic[0], '__dict__')
    with pytest.raises(FrozenInstanceError):
        meta.periodic = ()
    with pytest.raises(FrozenInstanceError):
        meta.periodic[0].periodic = True
    assert hash(meta) == hash(pickle.loads(pickle.dumps(meta)))

    pickle_path = tmp_path / 'transform_meta.pkl'
    meta.save(str(pickle_path))
    assert TransformMeta.load(str(pickle_path)) == meta

    wire_value = json.loads(json.dumps(meta.to_json()))
    assert TransformMeta.from_json(wire_value) == meta
    assert TransformMeta.from_dict(meta.to_dict()) == meta


def test_transform_meta_rejects_invalid_or_noncanonical_records():
    first = PeriodicEntry('U', (), 'z', (), False)
    second = PeriodicEntry('U', (), 'a', (), True)

    with pytest.raises(ValueError, match='canonical'):
        TransformMeta(periodic=(first, second))
    with pytest.raises(ValueError, match='unique'):
        TransformMeta(periodic=(first, first))
    with pytest.raises(ValueError, match='schema version'):
        TransformMeta.from_dict({'schema_version': 2, 'periodic': []})
    with pytest.raises(ValueError, match='schema version'):
        TransformMeta.from_dict({'schema_version': True, 'periodic': []})


def test_transform_meta_changes_jit_static_input_identity():
    trace_count = 0

    @jax.jit
    def use_meta(meta):
        nonlocal trace_count
        trace_count += 1
        return jnp.asarray(len(meta.periodic), dtype=jnp.int32)

    empty = TransformMeta()
    periodic = TransformMeta(periodic=(
        PeriodicEntry('U', (), 'phase', (), True),
    ))

    assert use_meta(empty) == 0
    assert use_meta(empty) == 0
    assert use_meta(periodic) == 1
    assert trace_count == 2


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
