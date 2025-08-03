import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, List

import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import lax
from jax import numpy as jnp

from jaxctx import wrap_random, get_parameter
from jaxctx.priors.types import FloatArray, IntArray, BoolArray, ComplexArray

tfpd = tfp.distributions

CoDomain = ComplexArray | FloatArray | IntArray | BoolArray


class AbstractPrior(ABC):
    """
    Represents a generative prior.
    """

    def __init__(self, name: str | None, base_dtype):
        self._name = name
        self._base_dtype = base_dtype

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
    def base_dtype(self):
        """
        The dtype of the prior random variable in X-space.
        """
        return self._base_dtype

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

    def parameter(self, random_init=False, param_collection: str = 'params', rng_stream: str = 'params'):
        """
        Convert a prior into a constrained parameter, that takes a single value in the model, but still has an associated
        log_prob. The parameter is registered into the corresponding collection and stream.

        Args:
            random_init: Whether to initialise the parameter randomly or at the median of the distribution.
            param_collection: The collection to register the parameter in.
            rng_stream: The name of the random number generator stream to use for sampling.

        Returns:
            a parameter constrained to the prior distribution.
        """
        return prior_to_parameter(prior=self, random_init=random_init, param_collection=param_collection,
                                  rng_stream=rng_stream)

    def realise(self, U_collection: str = 'U', X_collection: str = 'X', rng_stream: str = 'U'):
        """
        Realise the prior distribution into a parameter.

        Args:
            U_collection: The collection to register the parameter in.
            X_collection: The collection to register the parameter in.
            rng_stream: The name of the random number generator stream to use for sampling U.

        Returns:
            A parameter representing the prior.
        """
        return realise_prior(prior=self, U_collection=U_collection, X_collection=X_collection, rng_stream=rng_stream)


class Prior(AbstractPrior):
    """
    Represents a generative prior.
    """

    def __init__(self, dist: tfpd.Distribution, name: Optional[str] = None, base_dtype=jnp.float32):
        AbstractPrior.__init__(self, name=name, base_dtype=base_dtype)
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
    return 0.5 * (x / (1 + lax.abs(x)) + 1)


def quick_unit_inverse(y: jax.Array) -> jax.Array:
    """
    Inverse of quick_unit.

    Args:
        y: jax.Array value in (0, 1) open interval

    Returns:
        value in (-inf, inf) in open interval
    """
    twoy = y + y

    return jnp.where(
        y >= 0.5,
        (1 - twoy) / (twoy - 2),
        1 - lax.reciprocal(twoy)
    )


def sample_quick_unit(key, shape, dtype):
    """
    Sample from the quick unit distribution.

    Args:
        key: PRNGKey to use.
        shape: Shape of the output.
        dtype: Dtype of the output.

    Returns:
        A jax.Array sampled from the quick unit distribution.
    """

    def normal_cdf(x):
        # CDF of normal distribution using scipy's erf function
        return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

    return quick_unit_inverse(normal_cdf(jax.random.normal(key, shape, dtype)))


def prior_to_parameter(prior: AbstractPrior, random_init: bool, param_collection: str = 'params',
                       X_collection: str = 'X', log_prob_collection: str = 'log_prob', rng_stream: str = 'params'):
    """
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
    if prior.name is None:
        raise ValueError("Prior must have a name to be parametrised.")
    # Initialises at median of distribution using zeros, else unit-normal.

    if random_init:
        initaliser = wrap_random(sample_quick_unit, rng_stream)
    else:
        initaliser = jnp.zeros
    if prior.base_ndims == 0:
        warnings.warn(f"Creating a zero-sized parameter for {prior.name}. Probably unintended.")
    norm_U_base_param = get_parameter(
        name=prior.name,
        shape=prior.base_shape,
        dtype=prior.base_dtype,
        init=initaliser,
        collection=param_collection
    )
    # transform [-inf, inf] -> [0,1]
    U_base_param = quick_unit(norm_U_base_param)
    X_param = get_parameter(
        name=prior.name,
        collection=X_collection,
        init=prior.forward(U_base_param)
    )
    # Register the log_prob as a parameter
    _ = get_parameter(
        name=prior.name,
        collection=log_prob_collection,
        init=prior.log_prob(X_param)
    )
    return X_param


def realise_prior(prior: AbstractPrior, U_collection: str = 'U', X_collection: str = 'X',
                  log_prob_collection: str = 'log_prob', rng_stream: str = 'U'):
    """
    Convert a prior into a non-Bayesian parameter, that takes a single value in the model, but still has an associated
    log_prob. The parameter is registered as a `jaxns.get_parameter` with added `_param` name suffix.

    To constrain the parameter we use a Normal parameter with centre on unit cube, and scale covering the whole cube,
    as the base representation. This base representation covers the whole real line and be reliably used with SGD, etc.

    Args:
        prior: any prior
        U_collection: the collection to register the parameter in for U-space.
        X_collection: the collection to register the parameter in for X-space.
        rng_stream: the name of the random number generator stream to use for sampling U.

    Returns:
        A parameter representing the prior.
    """
    if prior.name is None:
        raise ValueError("Prior must have a name to be parametrised.")
    # Initialises at median of distribution using zeros, else unit-normal.
    initaliser = wrap_random(jax.random.uniform, rng_stream)
    if prior.base_ndims == 0:
        warnings.warn(f"Creating a zero-sized parameter for {prior.name}. Probably unintended.")
    U_base_param = get_parameter(
        name=prior.name,
        shape=prior.base_shape,
        dtype=prior.base_dtype,
        init=initaliser,
        collection=U_collection
    )
    X_param = get_parameter(
        name=prior.name,
        collection=X_collection,
        init=prior.forward(U_base_param)
    )
    # Register the log_prob as a parameter
    _ = get_parameter(
        name=prior.name,
        collection=log_prob_collection,
        init=prior.log_prob(X_param)
    )
    return X_param
