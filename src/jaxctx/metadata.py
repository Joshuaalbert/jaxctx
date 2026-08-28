import dataclasses
from typing import Any

from jaxctx.pytree import Pytree

__all__ = [
    'PeriodicEntry',
    'TransformMeta',
]


_TRANSFORM_META_SCHEMA_VERSION = 1


@dataclasses.dataclass(slots=True, frozen=True)
class PeriodicEntry:
    """Static topology declaration for one realised prior's U coordinates."""

    collection: str
    scope: tuple[str, ...]
    name: str
    base_shape: tuple[int, ...]
    periodic: bool

    def __post_init__(self):
        if type(self.collection) is not str:
            raise TypeError(f"collection must be a str, got {type(self.collection).__name__}.")
        if type(self.scope) is not tuple or any(type(value) is not str for value in self.scope):
            raise TypeError("scope must be a tuple of str values.")
        if type(self.name) is not str:
            raise TypeError(f"name must be a str, got {type(self.name).__name__}.")
        if type(self.base_shape) is not tuple or any(
            type(value) is not int or value < 0 for value in self.base_shape
        ):
            raise TypeError("base_shape must be a tuple of non-negative int values.")
        if type(self.periodic) is not bool:
            raise TypeError(f"periodic must be a bool, got {type(self.periodic).__name__}.")


@dataclasses.dataclass(slots=True, frozen=True)
class TransformMeta(Pytree):
    """Immutable static metadata discovered while a transformed model is initialised."""

    periodic: tuple[PeriodicEntry, ...] = ()

    def __post_init__(self):
        if type(self.periodic) is not tuple or any(
            type(entry) is not PeriodicEntry for entry in self.periodic
        ):
            raise TypeError("periodic must be a tuple of PeriodicEntry values.")
        keys = tuple((entry.collection, entry.scope, entry.name) for entry in self.periodic)
        if keys != tuple(sorted(keys)):
            raise ValueError("periodic entries must be in canonical collection, scope, and name order.")
        if len(keys) != len(set(keys)):
            raise ValueError("periodic entries must have unique collection, scope, and name keys.")

    @classmethod
    def flatten(cls, this):
        # The whole metadata value is static topology: it must not create array leaves.
        return cls.build_flatten(this, ['periodic'])

    @classmethod
    def unflatten(cls, aux_data, children):
        return cls.build_unflatten(aux_data, children)

    def to_json(self) -> dict[str, Any]:
        """Return the versioned JSON-safe metadata representation."""
        return {
            'schema_version': _TRANSFORM_META_SCHEMA_VERSION,
            'periodic': [
                {
                    'collection': entry.collection,
                    'scope': list(entry.scope),
                    'name': entry.name,
                    'base_shape': list(entry.base_shape),
                    'periodic': entry.periodic,
                }
                for entry in self.periodic
            ],
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> 'TransformMeta':
        """Restore metadata from its versioned JSON-safe representation."""
        if type(value) is not dict or set(value) != {'schema_version', 'periodic'}:
            raise ValueError("TransformMeta data must contain exactly schema_version and periodic.")
        if (
            type(value['schema_version']) is not int
            or value['schema_version'] != _TRANSFORM_META_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported TransformMeta schema version {value['schema_version']!r}; "
                f"expected {_TRANSFORM_META_SCHEMA_VERSION}."
            )
        if type(value['periodic']) is not list:
            raise TypeError("TransformMeta periodic data must be a list.")

        entries = []
        entry_fields = {'collection', 'scope', 'name', 'base_shape', 'periodic'}
        for entry_value in value['periodic']:
            if type(entry_value) is not dict or set(entry_value) != entry_fields:
                raise ValueError(f"Periodic entry must contain exactly {sorted(entry_fields)}.")
            if type(entry_value['scope']) is not list:
                raise TypeError("Periodic entry scope must be a list.")
            if type(entry_value['base_shape']) is not list:
                raise TypeError("Periodic entry base_shape must be a list.")
            entries.append(
                PeriodicEntry(
                    collection=entry_value['collection'],
                    scope=tuple(entry_value['scope']),
                    name=entry_value['name'],
                    base_shape=tuple(entry_value['base_shape']),
                    periodic=entry_value['periodic'],
                )
            )
        entries.sort(key=lambda entry: (entry.collection, entry.scope, entry.name))
        return cls(periodic=tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_json` for dictionary-oriented callers."""
        return self.to_json()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'TransformMeta':
        """Alias for :meth:`from_json` for dictionary-oriented callers."""
        return cls.from_json(value)


TransformMeta.register_pytree()
