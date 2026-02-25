import jax
import numpy as np
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
