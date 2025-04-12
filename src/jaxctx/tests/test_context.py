import jax
import numpy as np
from jax import numpy as jnp

from jaxctx.context import get_parameter, wrap_random, set_parameter, transform, ScopedDict, next_rng_key, TransformedFn


def test_transform():
    with jax.checking_leaks():
        def f(x) -> jax.Array:
            y = get_parameter(
                'y', 'params', (), jnp.float32,
                init=wrap_random(jax.random.normal, 'params')
            )
            s = get_parameter('s', 'state', y.shape, y.dtype, init=jnp.zeros)
            s = set_parameter('s', 'state', s + x + y)
            return (x + y) / s

        transformed = transform(f)

        init = jax.jit(transformed.init)({'params': jax.random.PRNGKey(0)}, {}, 1)
        print(init)

        response = jax.jit(transformed.apply)({'params': jax.random.PRNGKey(0)}, init.collections, 1)
        print(response)
        assert response.fn_val == 1
        for key, val in response.collections.items():
            assert isinstance(val, ScopedDict)


def lift_scan(body):
    body_transformed = transform(body)

    def body_lifted(joint_carry, x):
        (rngs, collections, carry) = joint_carry
        two_rngs = jax.tree.map(lambda key: jax.random.split(key, 2), rngs)
        rngs = jax.tree_map(lambda key: key[0], two_rngs)
        rngs_body = jax.tree_map(lambda key: key[1], two_rngs)
        apply_return = body_transformed.apply(rngs_body, collections, carry, x)
        carry, y = apply_return.fn_val
        collections = apply_return.collections
        return (rngs, collections, carry), y

    def init(body_rngs, body_collections, carry, xs):
        scan_params = body_transformed.init(body_rngs, body_collections, carry, xs)
        return scan_params

    def apply(body_rngs, body_collections, carry, xs):
        init_lifted = (body_rngs, body_collections, carry)
        (rngs, collections, carry), ys = jax.lax.scan(body_lifted, init_lifted, xs)
        return carry, ys

    return TransformedFn(_init_fn=init, _apply_fn=apply)


def test_transform_scan():
    def mlp(x) -> jax.Array:
        W = get_parameter('W', 'params', (np.shape(x)[-1], np.shape(x)[-1]), jnp.float32,
                          init=wrap_random(jax.random.normal, 'params'))
        b = get_parameter('b', 'params', (np.shape(x)[-1],), jnp.float32, init=jnp.zeros)
        return jnp.dot(W, x) + b

    def f(carry, xs):
        # recurrent application on input
        def body(carry, x):
            out = mlp(carry) + carry
            return out, ()

        transformed_scan = lift_scan(body)
        return transformed_scan.apply(carry, xs)

    transformed = transform(f)

    carry = jnp.ones((5,))
    xs = jnp.arange(10)

    init = transformed.init({'scan_rng': jax.random.PRNGKey(0)}, {}, carry, xs)
    print(init)

    response = transformed.apply({'scan_rng': jax.random.PRNGKey(0)}, init.collections, carry, xs)
    print(response)
