import dataclasses
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import jax
from jax import numpy as jnp
from jax._src.typing import SupportsDType

from jaxctx.metadata import PeriodicEntry, TransformMeta
from jaxctx.priors.types import PRNGKey
from jaxctx.pytree import PureDataclassPytree

__all__ = [
    'PeriodicEntry',
    'TransformMeta',
    'get_parameter',
    'set_parameter',
    'transform',
    'convert_external_params',
    'wrap_random',
    'next_rng_key',
    'get_base_dtype',
    'scope',
    'CtxParams',
    'ScopedDict'
]


class ScopedDict:
    """
    Nested dictionary keyed by scopes. Non-terminal nodes are scopes (dicts),
    terminal nodes are values.
    """

    def __init__(self, _dict=None, _scopes=None):
        self._scopes: List[str] = [] if _scopes is None else _scopes
        self._dict = {} if _dict is None else _dict

    def with_scopes(self, scopes: List[str]) -> 'ScopedDict':
        return ScopedDict(_dict=self._dict, _scopes=list(scopes))

    @property
    def scope_prefix(self):
        return '.'.join(self._scopes)

    def to_dict(self):
        return self._dict

    def __repr__(self):
        return f"ScopedDict(scopes={repr(self._scopes)}, dict={repr(self._dict)})"

    def _get_scope_dict(self, create: bool = False) -> Optional[Dict[str, Any]]:
        node = self._dict
        for scope in self._scopes:
            if scope not in node:
                if not create:
                    return None
                node[scope] = {}
            child = node[scope]
            if not isinstance(child, dict):
                raise ValueError(f"Scope '{scope}' collides with existing leaf.")
            node = child
        return node

    @staticmethod
    def _split_dotted(dotted_key: str) -> List[str]:
        if dotted_key.startswith('.'):
            dotted_key = dotted_key[1:]
        if dotted_key == "":
            return []
        parts = dotted_key.split('.')
        if any(part == "" for part in parts):
            raise ValueError("Dotted key contains empty scope segment.")
        return parts

    def get_dotted(self, dotted_key: str):
        """
        Lookup a value from the root using dotted notation.
        Leading '.' is optional and ignored.
        """
        parts = self._split_dotted(dotted_key)
        node = self._dict
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(dotted_key)
            node = node[part]
        return node

    def set_dotted(self, dotted_key: str, value):
        """
        Set a value from the root using dotted notation.
        Leading '.' is optional and ignored.
        """
        parts = self._split_dotted(dotted_key)
        if not parts:
            raise ValueError("Dotted key cannot be empty.")
        node = self._dict
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            child = node[part]
            if not isinstance(child, dict):
                raise ValueError(f"Scope '{part}' collides with existing leaf.")
            node = child
        last_part = parts[-1]
        if last_part in node:
            existing = node[last_part]
            if isinstance(existing, dict) and not isinstance(value, dict):
                raise ValueError(f"Cannot overwrite scope '{last_part}' with a leaf value.")
            if not isinstance(existing, dict) and isinstance(value, dict):
                raise ValueError(f"Cannot overwrite leaf '{last_part}' with a scope.")
        node[last_part] = value

    def iter_items(self):
        """
        Iterate over leaf values yielding (dotted_key, value).
        Keys are returned relative to the root, prefixed by current scopes.
        """
        scope_dict = self._get_scope_dict(create=False)
        if scope_dict is None:
            return iter(())

        prefix = list(self._scopes)

        def _walk(node: Dict[str, Any], path_prefix: List[str]):
            for key in sorted(node.keys()):
                value = node[key]
                if isinstance(value, dict):
                    yield from _walk(value, path_prefix + [key])
                else:
                    dotted_key = '.'.join(path_prefix + [key])
                    yield dotted_key, value

        return _walk(scope_dict, prefix)

    def __getitem__(self, item):
        scope_dict = self._get_scope_dict(create=False)
        if scope_dict is None:
            raise KeyError(item)
        return scope_dict[item]

    def __setitem__(self, key, value):
        scope_dict = self._get_scope_dict(create=True)
        if scope_dict is None:
            raise ValueError("Scope path could not be resolved.")
        if key in scope_dict:
            existing = scope_dict[key]
            if isinstance(existing, dict) and not isinstance(value, dict):
                raise ValueError(f"Cannot overwrite scope '{key}' with a leaf value.")
            if not isinstance(existing, dict) and isinstance(value, dict):
                raise ValueError(f"Cannot overwrite leaf '{key}' with a scope.")
        scope_dict[key] = value

    def __contains__(self, item):
        scope_dict = self._get_scope_dict(create=False)
        if scope_dict is None:
            return False
        return item in scope_dict

    def __iter__(self):
        scope_dict = self._get_scope_dict(create=False)
        if scope_dict is None:
            return iter(())
        return iter(scope_dict)

    def __len__(self):
        scope_dict = self._get_scope_dict(create=False)
        if scope_dict is None:
            return 0
        return len(scope_dict)

    def keys(self):
        scope_dict = self._get_scope_dict(create=False) or {}
        return scope_dict.keys()

    def values(self):
        scope_dict = self._get_scope_dict(create=False) or {}
        return scope_dict.values()

    def items(self):
        scope_dict = self._get_scope_dict(create=False) or {}
        return scope_dict.items()


# Add as pytree type

def _flatten_mapping(mapping: Dict[str, Any]) -> Tuple[List[Any], Tuple[Tuple[str, Any], ...]]:
    children: List[Any] = []
    structure: List[Tuple[str, Any]] = []
    for key in sorted(mapping.keys()):
        value = mapping[key]
        if isinstance(value, dict):
            sub_children, sub_structure = _flatten_mapping(value)
            children.extend(sub_children)
            structure.append((key, sub_structure))
        else:
            children.append(value)
            structure.append((key, None))
    return children, tuple(structure)


def _unflatten_mapping(structure: Tuple[Tuple[str, Any], ...], children_iter: Iterator[Any]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for key, sub_structure in structure:
        if sub_structure is None:
            mapping[key] = next(children_iter)
        else:
            mapping[key] = _unflatten_mapping(sub_structure, children_iter)
    return mapping


def scoped_dict_flatten(scoped_dict: ScopedDict):
    children, structure = _flatten_mapping(scoped_dict._dict)
    return (
        children,
        (structure, tuple(scoped_dict._scopes))
    )


def scoped_dict_unflatten(aux_data, children):
    structure, scopes = aux_data
    children_iter = iter(children)
    _dict = _unflatten_mapping(structure, children_iter)
    return ScopedDict(_dict=_dict, _scopes=list(scopes))


jax.tree_util.register_pytree_node(
    ScopedDict,
    scoped_dict_flatten,
    scoped_dict_unflatten
)

CtxParams = ScopedDict


class Ctx:
    def __init__(self, rngs: Dict[str, PRNGKey], collections: Dict[str, ScopedDict], stack: List['Ctx'], init: bool,
                 base_dtype):
        self._collections = defaultdict(ScopedDict)
        for key, val in collections.items():
            self._collections[key] = val
        self._rngs = rngs
        self._stack = stack
        self._scopes = []
        self._init = init
        self._base_dtype = base_dtype
        self._periodic_entries: Dict[Tuple[str, Tuple[str, ...], str], PeriodicEntry] = {}

    def next_rng_key(self, rng_stream: str) -> PRNGKey:
        if rng_stream not in self._rngs:
            raise ValueError(f"RNG stream {rng_stream} not provided.")
        self._rngs[rng_stream], new_rng = jax.random.split(self._rngs[rng_stream])
        return new_rng

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.remove(self)
        return

    @property
    def is_init(self) -> bool:
        return self._init

    @property
    def collections(self) -> Dict[str, ScopedDict]:
        return dict(self._collections.items())

    @property
    def base_dtype(self):
        return self._base_dtype

    def get_collection(self, collection: str) -> ScopedDict:
        return self._collections[collection].with_scopes(self._scopes)

    @property
    def transform_meta(self) -> TransformMeta:
        entries = tuple(sorted(
            self._periodic_entries.values(),
            key=lambda entry: (entry.collection, entry.scope, entry.name),
        ))
        return TransformMeta(periodic=entries)

    def record_periodic(self, *, collection: str, name: str, base_shape: Tuple[int, ...], periodic: bool):
        if not self._init:
            return
        entry = PeriodicEntry(
            collection=collection,
            scope=tuple(self._scopes),
            name=name,
            base_shape=tuple(int(size) for size in base_shape),
            periodic=periodic,
        )
        key = (entry.collection, entry.scope, entry.name)
        previous = self._periodic_entries.get(key)
        if previous is not None and previous != entry:
            qualified_name = '.'.join(entry.scope + (entry.name,))
            raise ValueError(
                f"Conflicting periodic declarations for {entry.collection}[{qualified_name!r}]: "
                f"first base_shape={previous.base_shape}, periodic={previous.periodic}; "
                f"second base_shape={entry.base_shape}, periodic={entry.periodic}."
            )
        self._periodic_entries[key] = entry

    def push_scope(self, scope):
        self._scopes.append(scope)

    def pop_scope(self):
        self._scopes.pop()


class GlobalContext:
    def __init__(self, rng: Optional[jax.Array] = None):
        self.stack: List[Ctx] = []

    def new(self, rngs: Dict[str, PRNGKey], collections: Dict[str, ScopedDict], init: bool, base_dtype):
        new_ctx = Ctx(rngs=rngs, collections=collections, stack=self.stack, init=init, base_dtype=base_dtype)
        self.stack.append(new_ctx)
        return new_ctx

    @property
    def collections(self) -> Dict[str, ScopedDict]:
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        return self.stack[-1].collections

    @property
    def base_dtype(self):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        return self.stack[-1].base_dtype

    @property
    def is_init(self) -> bool:
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        return self.stack[-1].is_init

    def get_collection(self, collection: str) -> ScopedDict:
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        return self.stack[-1].get_collection(collection)

    @property
    def transform_meta(self) -> TransformMeta:
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        return self.stack[-1].transform_meta

    def record_periodic(self, *, collection: str, name: str, base_shape: Tuple[int, ...], periodic: bool):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use `transform` to create a context.")
        self.stack[-1].record_periodic(
            collection=collection,
            name=name,
            base_shape=base_shape,
            periodic=periodic,
        )

    def next_rng_key(self, rng_stream: str) -> PRNGKey:
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        return self.stack[-1].next_rng_key(rng_stream)

    def push_scope(self, scope: str):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        self.stack[-1].push_scope(scope)

    def pop_scope(self):
        if len(self.stack) == 0:
            raise ValueError("No context available. Must use transform_with_state to create a context.")
        self.stack[-1].pop_scope()


global_context = GlobalContext()

PT = TypeVar('PT')
InitType = Union[jax.Array, Callable[[Tuple[int, ...], SupportsDType], PT], Callable[[], PT]]


@contextmanager
def scope(name: str):
    """
    Create a new scope, to prefix parameters and states, as {current_scope}.{name}.{param_name}.

    Args:
        name: the name of the scope

    Returns:
        The scope
    """
    # Context manager
    global_context.push_scope(name)
    try:
        yield
    finally:
        global_context.pop_scope()


def default_init(shape: Tuple[int, ...], dtype: SupportsDType):
    raise NotImplementedError("No init provided.")


def get_parameter(name: str, collection: str, shape: Optional[Tuple[int, ...]] = None,
                  dtype: Optional[SupportsDType] = None, *,
                  init: InitType = default_init) -> PT:
    """
    Get a parameter variable, initialised to a particular value. The parameter is stored in the given collection.

    Args:
        name: the name of the parameter
        collection: the collection to store parameter under.
        shape: the shape of the parameter must be provided if init is not a jax.Array
        dtype: the dtype of the parameter must be provided if init is not a jax.Array
        init: the initializer

    Returns:
        The parameter variable as a jax.Array
    """
    params = global_context.get_collection(collection)
    if name not in params:
        if callable(init):
            if (shape is None) and (dtype is None):
                new_param = init()
            else:
                new_param = init(shape, dtype)
        else:
            # Beware of closure issues.
            new_param = init
        params[name] = new_param

    return params[name]


def set_parameter(name: str, collection: str, value: PT) -> PT:
    """
    Set a variable in a collection to a particular value.
    If the name is not found in the collection then an error is raised.

    Args:
        name: the name of the state
        collection: the collection of the parameter
        value: the value to set

    Returns:
        The state variable as a jax.Array
    """
    params = global_context.get_collection(collection)
    if name not in params:
        raise ValueError(f"Parameter {name} not found. It must be initialised with `get_parameter` before it can be set.")
    # Ensure same pytree def
    tree_def = jax.tree.structure(params[name])
    value_tree_def = jax.tree.structure(value)
    if tree_def != value_tree_def:
        raise ValueError(f"Expected state with tree_def {tree_def} got {value_tree_def}.")
    if global_context.is_init:
        return params[name]
    params[name] = value
    return value


def set_state(name: str, collection: str, value: PT) -> PT:
    """
    Set a variable in a collection to a particular value.

    Args:
        name: the name of the state
        collection: the collection of the parameter
        value: the value to set

    Returns:
        The state variable as a jax.Array
    """
    params = global_context.get_collection(collection)
    if name in params:
        # Ensure same pytree def
        tree_def = jax.tree.structure(params[name])
        value_tree_def = jax.tree.structure(value)
        if tree_def != value_tree_def:
            raise ValueError(f"Expected state with tree_def {tree_def} got {value_tree_def}.")
    params[name] = value


def get_state(name: str, collection: str) -> PT:
    """
    Get a state in a collection.

    Args:
        name: the name of the state
        collection: the collection of the parameter

    Returns:
        The state variable as a jax.Array
    """
    params = global_context.get_collection(collection)
    if name not in params:
        raise ValueError(f"State {name} not found. It must be initialised before it can be set.")
    return params[name]


ExtParam = TypeVar('ExtParam')


def convert_external_params(external_params: ExtParam, collection: str, prefix: str = "") -> ExtParam:
    """
    Convert external parameters to context parameters. This can be used to convert haiku or flax parameters to
    jaxctx parameters.

    Args:
        external_params: map of name -> value
        collection: which collection to store the parameters in
        prefix: an optional prefix for the param name to make it unique.

    Returns:
        The context parameters
    """
    leaf_list, tree_def = jax.tree.flatten(external_params)

    def _unique_name(idx):
        return f"__{prefix}_{idx}"

    ctx_params = [get_parameter(_unique_name(idx), collection, init=leaf) for idx, leaf in enumerate(leaf_list)]
    external_params_ctx = jax.tree_unflatten(tree_def, ctx_params)
    return external_params_ctx


def wrap_random(f, rng_stream: str, **kwargs):
    """
    Wrap a function to use a random number generator from the context.
    Converts a function of (key, shape, dtype, **kwargs) into (shape, dtype).

    Args:
        f: the function to wrap, the args should be (key, shape, dtype, **kwargs)
        rng_stream: the rng stream to get from
        **kwargs options to pass to random generator.

    Returns:
        The wrapped function
    """

    @wraps(f)
    def wrapped(shape: Tuple[int, ...], dtype: SupportsDType):
        rng = next_rng_key(rng_stream)
        return f(rng, shape, dtype, **kwargs)

    return wrapped


FV = TypeVar('FV')


@dataclasses.dataclass(slots=True, frozen=True)
class ApplyReturn(PureDataclassPytree, Generic[FV]):
    fn_val: FV
    collections: Dict[str, ScopedDict]

    def __iter__(self):
        return iter((self.fn_val, self.collections))

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return (self.fn_val, self.collections)[item]


@dataclasses.dataclass(slots=True, frozen=True)
class InitReturn(PureDataclassPytree, Generic[FV]):
    """Numerical collections and init-only static transform metadata."""

    collections: Dict[str, ScopedDict]
    meta: TransformMeta = TransformMeta()

    def __iter__(self):
        return iter((self.collections, self.meta))

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return (self.collections, self.meta)[item]


ApplyReturn.register_pytree()
InitReturn.register_pytree()


@dataclasses.dataclass(slots=True, frozen=True)
class TransformedFn(Generic[FV]):
    _apply_fn: Callable
    _init_fn: Callable
    base_dtype: Any

    def apply(self, rngs, collections, *args, **kwargs) -> ApplyReturn[FV]:
        """
        Apply the function with given parameters and states.

        Args:
            rngs: a map of stream name to rng, e.g. {'params', jax.random.PRNGKey(0)}
            collections: a map of collection name to ScopedDict, e.g. {'params': ScopedDict()}
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            The function value and updated numerical collections.
        """
        return self._apply_fn(rngs, collections, *args, **kwargs)

    def init(self, rngs, collections, *args, **kwargs) -> InitReturn[FV]:
        """
        Initialise the transform.

        Args:
            rngs: a map of stream name to rng, e.g. {'params', jax.random.PRNGKey(0)}
            collections: a map of collection name to ScopedDict, e.g. {'params': ScopedDict()}
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            Initialised numerical collections and static transform metadata.
        """
        return self._init_fn(rngs, collections, *args, **kwargs)


def transform(f: Callable[..., FV], *, base_dtype=jnp.float32) -> TransformedFn[FV]:
    """
    Transform a function that uses parameters and states into a pure function.

    Args:
        f: the function to transform
        base_dtype: shared dtype for every unconstrained N value and unit-hypercube U value in the transformed model.

    Returns:
        The transformed init/apply functions and their shared base dtype.
    """
    base_dtype = jax.dtypes.canonicalize_dtype(base_dtype)
    if not jnp.issubdtype(base_dtype, jnp.floating):
        raise TypeError(f"base_dtype must be a floating dtype, got {base_dtype}.")

    def apply_fn(rngs, collections, *args, **kwargs) -> ApplyReturn[FV]:
        """
        Apply the function with given parameters and states.

        Args:
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            The function value and updated numerical collections.
        """
        if collections is None:
            collections = {}

        with global_context.new(rngs=rngs, collections=collections, init=False, base_dtype=base_dtype):
            fn_val = f(*args, **kwargs)
            return ApplyReturn(fn_val=fn_val, collections=global_context.collections)

    def init_fn(rngs, collections, *args, **kwargs) -> InitReturn[FV]:
        """
        Initialise the transform.

        Args:
            *args: args to function
            **kwargs: kwargs to function

        Returns:
            Initialised numerical collections and static transform metadata.
        """
        if collections is None:
            collections = {}

        with global_context.new(rngs=rngs, collections=collections, init=True, base_dtype=base_dtype):
            # can be sped up with aeval
            _ = f(*args, **kwargs)
            del _  # Ensure no closure issues
            return InitReturn(
                collections=global_context.collections,
                meta=global_context.transform_meta,
            )

    return TransformedFn(_init_fn=init_fn, _apply_fn=apply_fn, base_dtype=base_dtype)


def next_rng_key(rng_stream: str):
    """
    Get the next random number generator

    Returns:
        The next random number generator
    """
    return global_context.next_rng_key(rng_stream)


def get_base_dtype():
    """Return the shared N-space and U-space dtype for the active transformed model."""
    return global_context.base_dtype


def _record_periodic_realisation(*, collection: str, name: str, base_shape: Tuple[int, ...], periodic: bool):
    """Record init-only prior topology without exposing it as numerical collection state."""
    global_context.record_periodic(
        collection=collection,
        name=name,
        base_shape=base_shape,
        periodic=periodic,
    )
