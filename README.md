# jaxctx

JAX Context for memoizationg collections of parameters, and handling sequences of random keys.
Based loosely on things like haiku, and flax, but not deep learning specific. No support for lifting things like
scan, etc. You must build these things on top of this if you want them.

Additional support added for probabilistic parameterisations based on priors.

# Change Log

25 Feb, 2026 -- 1.1.0 released with scoped dicts structure changes. Breaks backward compatibility with 1.0.x, but adds
support for nested contexts and more flexible scoping.

21 July, 2025 -- 1.0.3 released with support for `jaxctx.prior` and `jaxctx.prior.Prior`.

3 June, 2025 -- 1.0.2 prior constrained parameters released.

2 June, 2025 -- 1.0.1 released with context API.
