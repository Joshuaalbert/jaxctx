from typing import TypeAlias, Union

import jax
import numpy as np

__all__ = [
    'PRNGKey',
    'Array',
    'ComplexArray',
    'IntArray',
    'FloatArray',
    'BoolArray'
]

#: JAX pseudo-random number generator key.
PRNGKey: TypeAlias = jax.Array

#: JAX or NumPy array, without scalar values.
Array: TypeAlias = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
#: JAX or NumPy array, with complex scalar values.
ComplexArray: TypeAlias = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    complex,  # valid scalars
]

#: JAX or NumPy array, with floating scalar values.
FloatArray: TypeAlias = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
#: JAX or NumPy array, with integer scalar values.
IntArray: TypeAlias = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
#: JAX or NumPy array, with boolean scalar values.
BoolArray: TypeAlias = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]
