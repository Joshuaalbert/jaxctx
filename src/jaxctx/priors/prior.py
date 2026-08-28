import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from jaxctx import get_parameter, wrap_random
from jaxctx.context import (
    _record_periodic_realisation,
    get_base_dtype,
    global_context,
    set_state,
)
from jaxctx.priors.types import BoolArray, ComplexArray, FloatArray, IntArray

tfpd = tfp.distributions

QUICK_UNIT_SCALE = 2.2

CoDomain = ComplexArray | FloatArray | IntArray | BoolArray


class AbstractPrior(ABC):
    """
    Represents a generative prior.
    """

    def __init__(self, name: str | None):
        self._name = name

    def __repr__(self):
        return f"{self.name if self.name is not None else '*'}\t{self.base_shape} -> {self.shape} {self.dtype}"

    @property
    def name(self):
        """
        The name of the prior.
        """
        return self._name

    @abstractmethod
    def _dtype(self):
        """
        The dtype of the prior.
        """
        ...

    @abstractmethod
    def _base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the prior, in U-space.
        """
        ...

    @abstractmethod
    def _shape(self) -> Tuple[int, ...]:
        """
        The shape of the prior, in X-space.
        """
        ...

    @abstractmethod
    def _forward(self, U: FloatArray) -> CoDomain:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        ...

    @abstractmethod
    def _inverse(self, X: CoDomain) -> FloatArray:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        ...

    @abstractmethod
    def _log_prob(self, X: CoDomain) -> FloatArray:
        """
        The log probability of the prior.

        Args:
            X: X-space representation

        Returns:
            log probability of the prior
        """

        ...

    @property
    def dtype(self):
        """
        The dtype of the prior random variable in X-space.
        """
        return self._dtype()

    @property
    def base_shape(self) -> Tuple[int, ...]:
        """
        The base shape of the prior random variable in U-space.
        """
        return self._base_shape()

    @property
    def base_ndims(self):
        """
        The number of dimensions of the prior random variable in U-space.
        """
        return int(np.prod(self.base_shape))

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The shape of the prior random variable in X-space.
        """
        return self._shape()

    def forward(self, U: FloatArray) -> CoDomain:
        """
        The forward transformation from U-space to X-space.

        Args:
            U: U-space representation

        Returns:
            X-space representation
        """
        return self._forward(U)

    def inverse(self, X: CoDomain) -> FloatArray:
        """
        The inverse transformation from X-space to U-space.

        Args:
            X: X-space representation

        Returns:
            U-space representation
        """
        return self._inverse(X)

    def log_prob(self, X: CoDomain) -> FloatArray:
        """
        The log probability of the prior.

        Args:
            X: X-space representation

        Returns:
            log probability of the prior
        """
        log_prob = self._log_prob(X)
        if np.size(log_prob) > 1:
            log_prob = jnp.sum(log_prob)
        if log_prob.shape != ():
            log_prob = jax.lax.reshape(log_prob, ())
        return log_prob

    def parameter(self, *, init: FloatArray | Callable|None = None, random_init=False, param_collection: str = 'params',
                  U_collection: str = 'U', X_collection='X', log_prob_collection='log_prob',
                  rng_stream: str = 'params'):
        """
        Convert a prior into a constrained parameter, that takes a single value in the model, but still has an associated
        log_prob. The parameter is registered into the corresponding collection and stream.

        Intent is to use these for optimisation.

        Args:
            init: optional initial value or callable, giving the initial value in X-space.
            random_init: Whether to initialise the parameter randomly or at the median of the distribution.
            param_collection: The collection to register the parameter in.
            rng_stream: The name of the random number generator stream to use for sampling.

        The active ``transform`` supplies the shared dtype for the unconstrained N value and derived U value.

        Returns:
            a parameter constrained to the prior distribution.
        """
        return prior_to_parameter(prior=self, init=init, random_init=random_init,
                                  param_collection=param_collection,
                                  U_collection=U_collection, X_collection=X_collection,
                                  log_prob_collection=log_prob_collection,
                                  rng_stream=rng_stream)

    def realise(self, *, periodic: bool = False, U_collection: str = 'U', X_collection: str = 'X',
                log_prob_collection='log_prob', rng_stream: str = 'U'):
        """
        Realise the prior from a sampled homogeneous U-space value.

        Args:
            periodic: whether every coordinate in this prior's base U shape has equivalent endpoints. Partial
                periodicity within one prior is unsupported.
            U_collection: The collection to register the parameter in.
            X_collection: The collection to register the parameter in.
            log_prob_collection: The collection to register the prior log probability in.
            rng_stream: The name of the random number generator stream to use for sampling U.

        Returns:
            A parameter representing the prior.
        """
        return realise_prior(prior=self, periodic=periodic, U_collection=U_collection, X_collection=X_collection,
                             log_prob_collection=log_prob_collection, rng_stream=rng_stream)


class Prior(AbstractPrior):
    """
    Represents a generative prior.
    """

    def __init__(self, dist: tfpd.Distribution, name: Optional[str] = None):
        AbstractPrior.__init__(self, name=name)
        self._dist_chain = TFPDistributionChain(dist)
        self._dist = dist

    @property
    def dist(self) -> tfpd.Distribution:
        """
        The distribution of the prior.
        """
        return self._dist

    def _base_shape(self) -> Tuple[int, ...]:
        return self._dist_chain.base_shape()

    def _shape(self) -> Tuple[int, ...]:
        return self._dist_chain.shape()

    def _dtype(self):
        return self._dist_chain.dtype()

    def _forward(self, U: FloatArray) -> CoDomain:
        return self._dist_chain.forward(U)

    def _inverse(self, X: CoDomain) -> FloatArray:
        return self._dist_chain.inverse(X)

    def _log_prob(self, X: CoDomain) -> FloatArray:
        return self._dist_chain.log_prob(X=X)


def distribution_chain(dist: tfpd.Distribution) -> List[tfpd.TransformedDistribution | tfpd.Sample | tfpd.Distribution]:
    """
    Returns a list of distributions that make up the chain of distributions.

    Args:
        dist: A TFP distribution, transformed distribution or sample.

    Returns:
        A list of distributions.
    """
    chain = []
    while True:
        chain.append(dist)
        if isinstance(dist, tfpd.TransformedDistribution):
            dist = dist.distribution
            continue
        break
    # Must reverse the chain because the first distribution is the last in the chain.
    return chain[::-1]


class TFPDistributionChain:
    """
    Represents a wrapped TFP distribution.
    """

    def __init__(self, dist: tfpd.Distribution):
        self.dist_chain = distribution_chain(dist)
        check_dist = self.dist_chain[0]
        if isinstance(self.dist_chain[0], tfpd.Sample):
            check_dist = self.dist_chain[0].distribution
        if '_quantile' not in check_dist.__class__.__dict__:
            # TODO(Joshuaalbert): we could numerically approximate it. This requires knowing the support of dist.
            # Repartitioning the prior also requires knowing the support and choosing a replacement, which is not
            # always easy from stats. E.g. StudentT variance doesn't exist but a numerial quantile can be made.
            raise ValueError(f"Distribution {dist} is missing a quantile.")

    def __repr__(self):
        return " -> ".join(map(repr, self.dist_chain))

    def dtype(self):
        return self.dist_chain[-1].dtype

    def base_shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[0].batch_shape_tensor()) + tuple(self.dist_chain[0].event_shape_tensor())

    def shape(self) -> Tuple[int, ...]:
        return tuple(self.dist_chain[-1].batch_shape_tensor()) + tuple(self.dist_chain[-1].event_shape_tensor())

    def forward(self, U) -> Union[FloatArray, IntArray, BoolArray]:
        dist = self.dist_chain[0]
        # cast U to the correct dtype for the distribution
        U = U.astype(jnp.result_type(U.dtype, dist.dtype))
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.quantile(U)
        for dist in self.dist_chain[1:]:
            X = dist.bijector.forward(X)
        return X

    def inverse(self, X) -> FloatArray:
        for dist in reversed(self.dist_chain[1:]):
            X = dist.bijector.inverse(X)
        dist = self.dist_chain[0]
        if isinstance(dist, tfpd.Sample):
            dist = dist.distribution
        X = dist.cdf(X)
        return X

    def log_prob(self, X):
        return self.dist_chain[-1].log_prob(X)


def quick_unit(x: jax.Array) -> jax.Array:
    """
    Quick approximation to the sigmoid.

    Args:
        x: jax.Array value in (-inf, inf) open interval

    Returns:
        value in (0, 1) in open interval
    """
    return jax.nn.sigmoid(x / QUICK_UNIT_SCALE)


def quick_unit_inverse(y: jax.Array) -> jax.Array:
    """
    Inverse of quick_unit.

    Args:
        y: jax.Array value in (0, 1) open interval

    Returns:
        value in (-inf, inf) in open interval
    """
    return QUICK_UNIT_SCALE * jax.scipy.special.logit(y)


def sample_quick_unit_dist(key, shape, dtype):
    """
    Sample from the quick unit distribution in the param space, a logit-normal distribution with mean 0 and scale
    QUICK_UNIT_SCALE.

    Args:
        key: PRNGKey to use.
        shape: Shape of the output.
        dtype: Dtype of the output.

    Returns:
        A jax.Array sampled from the quick unit distribution.
    """

    return jax.random.normal(key, shape, dtype)


def _validate_parameter_prior(prior: AbstractPrior):
    if prior.name is None:
        raise ValueError("Prior must have a name to be parametrised.")
    if prior.base_ndims == 0:
        warnings.warn(f"Creating a zero-sized parameter for {prior.name}. Probably unintended.")


def _validate_shape_dtype(*, actual_shape, actual_dtype, shape: Tuple[int, ...], dtype, label: str):
    expected_shape = tuple(shape)
    if tuple(actual_shape) != expected_shape:
        raise ValueError(f"{label} has shape {actual_shape}, expected {expected_shape}.")
    expected_dtype = jax.dtypes.canonicalize_dtype(dtype)
    actual_dtype = jax.dtypes.canonicalize_dtype(actual_dtype)
    if actual_dtype != expected_dtype:
        raise ValueError(f"{label} has dtype {actual_dtype}, expected {expected_dtype}.")


def _validate_array_contract(value, *, shape: Tuple[int, ...], dtype, label: str):
    _validate_shape_dtype(
        actual_shape=value.shape,
        actual_dtype=value.dtype,
        shape=shape,
        dtype=dtype,
        label=label
    )


def _physical_to_unconstrained(prior: AbstractPrior, value, base_dtype, *, cast_physical_dtype: bool = False):
    X = jnp.asarray(value, dtype=prior.dtype if cast_physical_dtype else None)
    _validate_array_contract(
        X,
        shape=prior.shape,
        dtype=prior.dtype,
        label=f"Physical initial value for {prior.name}"
    )
    U = jnp.asarray(prior.inverse(X), dtype=base_dtype)
    _validate_array_contract(
        U,
        shape=prior.base_shape,
        dtype=base_dtype,
        label=f"Unit initial value for {prior.name}"
    )
    N = jnp.asarray(quick_unit_inverse(jnp.clip(U, 1e-6, 1 - 1e-6)), dtype=base_dtype)
    _validate_array_contract(
        N,
        shape=prior.base_shape,
        dtype=base_dtype,
        label=f"Unconstrained initial value for {prior.name}"
    )
    return N


def _parameter_initialiser(prior: AbstractPrior, init: FloatArray | Callable | None, random_init: bool,
                           rng_stream: str, base_dtype):
    if init is not None:
        # transform: X -> U -> N
        if callable(init):
            @partial(wrap_random, rng_stream=rng_stream)
            def initialiser(key, shape, dtype):
                _validate_shape_dtype(
                    actual_shape=shape,
                    actual_dtype=dtype,
                    shape=prior.base_shape,
                    dtype=base_dtype,
                    label=f"Requested unconstrained initial value for {prior.name}"
                )
                physical_dtype = jax.dtypes.canonicalize_dtype(prior.dtype)
                X = init(key, prior.shape, physical_dtype)
                return _physical_to_unconstrained(prior, X, base_dtype)
        else:
            def initialiser(shape, dtype):
                _validate_shape_dtype(
                    actual_shape=shape,
                    actual_dtype=dtype,
                    shape=prior.base_shape,
                    dtype=base_dtype,
                    label=f"Requested unconstrained initial value for {prior.name}"
                )
                return _physical_to_unconstrained(prior, init, base_dtype, cast_physical_dtype=True)
    elif random_init:
        initialiser = wrap_random(sample_quick_unit_dist, rng_stream)
    else:
        # Initialises at median of distribution using zeros.
        initialiser = jnp.zeros
    return initialiser


class ParameterPack:
    """
    Packs several prior-constrained parameters into one unconstrained parameter vector.

    The pack is the unit of optimisation: it has one trainable leaf and one transform from the real line to the unit
    hypercube. The active ``transform`` supplies the shared N/U dtype for packed and independent parameters alike.
    Unit-space slices are reshaped and passed through each prior independently.
    """

    def __init__(self, priors: Sequence[AbstractPrior], name: str = 'packed'):
        self._priors = tuple(priors)
        self._name = name
        if len(self._priors) == 0:
            raise ValueError("ParameterPack requires at least one prior.")
        if not isinstance(name, str) or name == '':
            raise ValueError("ParameterPack name must be a non-empty string.")
        for prior in self._priors:
            if not isinstance(prior, AbstractPrior):
                raise TypeError(f"Expected an AbstractPrior, got {type(prior)}.")
            _validate_parameter_prior(prior)

        names = tuple(prior.name for prior in self._priors)
        if len(names) != len(set(names)):
            raise ValueError(f"ParameterPack prior names must be unique, got {names}.")

        offsets = [0]
        for prior in self._priors:
            offsets.append(offsets[-1] + prior.base_ndims)
        self._offsets = tuple(offsets)

    @property
    def name(self) -> str:
        return self._name

    @property
    def priors(self) -> Tuple[AbstractPrior, ...]:
        return self._priors

    @property
    def base_ndims(self) -> int:
        return self._offsets[-1]

    def parameter(self, *, init: Optional[Sequence[FloatArray | Callable | None]] = None,
                  random_init: bool = False, param_collection: str = 'params', U_collection: str = 'U',
                  X_collection: str = 'X', log_prob_collection: str = 'log_prob',
                  rng_stream: str = 'params') -> Tuple[CoDomain, ...]:
        """
        Create physical parameters backed by one packed unconstrained parameter.

        Args:
            init: optional sequence of physical values or callables in X-space, one per prior. A callable receives the
                physical shape and dtype of its prior.
            random_init: whether members without an explicit initial value start from a standard normal draw.
            param_collection: collection containing the single packed trainable parameter.
            U_collection: collection containing the packed unit-hypercube value.
            X_collection: collection containing physical values under their prior names.
            log_prob_collection: collection containing per-prior log probabilities.
            rng_stream: random stream used by random or callable initialisers.

        Returns:
            Tuple of physical values in the same order as ``priors``.
        """
        base_dtype = get_base_dtype()
        if init is None:
            if random_init:
                packed_initialiser = wrap_random(sample_quick_unit_dist, rng_stream)
            else:
                packed_initialiser = jnp.zeros
        else:
            initial_values = tuple(init)
            if len(initial_values) != len(self._priors):
                raise ValueError(
                    f"ParameterPack {self.name} expected {len(self._priors)} initial values, "
                    f"got {len(initial_values)}."
                )

            initialisers = tuple(
                _parameter_initialiser(prior, initial_value, random_init, rng_stream, base_dtype)
                for prior, initial_value in zip(self._priors, initial_values)
            )

            def packed_initialiser(shape, dtype):
                chunks = []
                for prior, initialiser in zip(self._priors, initialisers):
                    chunk = initialiser(prior.base_shape, base_dtype)
                    _validate_array_contract(
                        chunk,
                        shape=prior.base_shape,
                        dtype=base_dtype,
                        label=f"Unconstrained initial value for {prior.name}"
                    )
                    chunks.append(jnp.reshape(chunk, (prior.base_ndims,)))
                packed = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks)
                _validate_array_contract(
                    packed,
                    shape=shape,
                    dtype=dtype,
                    label=f"Packed initial value for {self.name}"
                )
                return packed

        N_packed = get_parameter(
            name=self.name,
            shape=(self.base_ndims,),
            dtype=base_dtype,
            init=packed_initialiser,
            collection=param_collection
        )
        _validate_array_contract(
            N_packed,
            shape=(self.base_ndims,),
            dtype=base_dtype,
            label=f"Packed parameter {self.name}"
        )
        U_packed = quick_unit(N_packed)
        _validate_array_contract(
            U_packed,
            shape=(self.base_ndims,),
            dtype=base_dtype,
            label=f"Packed unit value {self.name}"
        )
        set_state(name=self.name, collection=U_collection, value=U_packed)

        physical_values = []
        for prior, start, stop in zip(self._priors, self._offsets[:-1], self._offsets[1:]):
            U = jax.lax.slice_in_dim(U_packed, start, stop)
            U = jax.lax.reshape(U, prior.base_shape)
            X = prior.forward(U)
            set_state(name=prior.name, collection=X_collection, value=X)
            set_state(name=prior.name, collection=log_prob_collection, value=prior.log_prob(X))
            physical_values.append(X)
        return tuple(physical_values)


def prior_to_parameter(prior: AbstractPrior, init: FloatArray | Callable | None = None, random_init: bool = False,
                       param_collection: str = 'params',
                       U_collection: str = 'U', X_collection: str = 'X', log_prob_collection: str = 'log_prob',
                       rng_stream: str = 'params'):
    """
    Creates a parameter from a prior transformed from a homogeneous unconstrained base measure.

    Convert a prior into a non-Bayesian parameter, that takes a single value in the model, but still has an associated
    log_prob. The parameter is registered as a `jaxns.get_parameter` with added `_param` name suffix.

    To constrain the parameter we use a Normal parameter with centre on unit cube, and scale covering the whole cube,
    as the base representation. This base representation covers the whole real line and be reliably used with SGD, etc.

    Args:
        prior: any prior
        random_init: whether to initialise the parameter randomly or at the median of the distribution.
        param_collection: the collection to register the parameter in.
        rng_stream: the name of the random number generator stream to use for sampling.

    Returns:
        A parameter representing the prior.
    """
    _validate_parameter_prior(prior)
    base_dtype = get_base_dtype()
    initialiser = _parameter_initialiser(prior, init, random_init, rng_stream, base_dtype)
    N_base_param = get_parameter(
        name=prior.name,
        shape=prior.base_shape,
        dtype=base_dtype,
        init=initialiser,
        collection=param_collection
    )
    _validate_array_contract(
        N_base_param,
        shape=prior.base_shape,
        dtype=base_dtype,
        label=f"Parameter {prior.name}"
    )
    # transform [-inf, inf] -> [0,1]
    U_base_param = quick_unit(N_base_param)
    _validate_array_contract(
        U_base_param,
        shape=prior.base_shape,
        dtype=base_dtype,
        label=f"Unit value {prior.name}"
    )
    set_state(name=prior.name, collection=U_collection, value=U_base_param)
    X_param = prior.forward(U_base_param)
    set_state(name=prior.name, collection=X_collection, value=X_param)
    set_state(name=prior.name, collection=log_prob_collection, value=prior.log_prob(X_param))
    return X_param


def realise_prior(prior: AbstractPrior, U_collection: str = 'U', X_collection: str = 'X',
                  log_prob_collection: str = 'log_prob', rng_stream: str = 'U', *, periodic: bool = False):
    """
    Realise a prior from U and record its static topology during transform initialisation.

    Args:
        prior: any prior
        U_collection: the collection to register the parameter in for U-space.
        X_collection: the collection to register the parameter in for X-space.
        log_prob_collection: the collection to register the prior log probability in.
        rng_stream: the name of the random number generator stream to use for sampling U.
        periodic: whether every coordinate in the prior's base U shape has equivalent endpoints. Partial periodicity
            within one prior is unsupported.

    Returns:
        A parameter representing the prior.
    """
    if prior.name is None:
        raise ValueError("Prior must have a name to be realised.")
    # Topology is a model-construction contract. Keep validation and collection
    # out of repeated apply tracing and non-jitted apply calls.
    if global_context.is_init:
        if type(periodic) is not bool:
            raise TypeError(
                f"periodic for prior {prior.name!r} must be a Python bool, "
                f"got {type(periodic).__name__}."
            )
        if periodic and not jnp.issubdtype(prior.dtype, jnp.inexact):
            raise TypeError(
                f"periodic=True requires a continuous prior output for {prior.name!r}, "
                f"got dtype {prior.dtype}. Cyclic discrete variables are unsupported."
            )
        _record_periodic_realisation(
            collection=U_collection,
            name=prior.name,
            base_shape=prior.base_shape,
            periodic=periodic,
        )
    # Draw U only when the collection entry is absent; apply reuses the supplied U.
    initializer = wrap_random(jax.random.uniform, rng_stream)
    if prior.base_ndims == 0:
        warnings.warn(f"Creating a zero-sized parameter for {prior.name}. Probably unintended.")
    base_dtype = get_base_dtype()
    U_base_param = get_parameter(
        name=prior.name,
        shape=prior.base_shape,
        dtype=base_dtype,
        init=initializer,
        collection=U_collection
    )
    _validate_array_contract(
        U_base_param,
        shape=prior.base_shape,
        dtype=base_dtype,
        label=f"Realised unit value {prior.name}"
    )
    X_param = prior.forward(U_base_param)
    set_state(name=prior.name, collection=X_collection, value=X_param)
    set_state(name=prior.name, collection=log_prob_collection, value=prior.log_prob(X_param))
    return X_param
