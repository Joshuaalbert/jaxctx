import importlib
from typing import get_args

import jax
import numpy as np

from jaxctx.priors import types


def test_type_alias_module_reloads_without_mutating_typing_objects():
    reloaded = importlib.reload(types)

    assert get_args(reloaded.Array) == (jax.Array, np.ndarray)
    assert get_args(reloaded.ComplexArray) == (jax.Array, np.ndarray, complex)
    assert get_args(reloaded.FloatArray) == (jax.Array, np.ndarray, float)
    assert get_args(reloaded.IntArray) == (jax.Array, np.ndarray, int)
    assert get_args(reloaded.BoolArray) == (jax.Array, np.ndarray, np.bool_, bool)
