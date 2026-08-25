import dataclasses
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any

import jax
import numpy as np

__all__ = [
    'Pytree',
    'PureDataclassPytree',
]


class Pytree(ABC):
    __slots__ = ()

    def size_info(self):
        def format_size_info(leaf):
            try:
                nbytes = leaf.nbytes
                if nbytes > 1024 ** 3:
                    memory = nbytes // 1024 ** 3
                    unit = "GB"
                elif nbytes > 1024 ** 2:
                    memory = nbytes // 1024 ** 2
                    unit = "MB"
                elif nbytes > 1024:
                    memory = nbytes // 1024
                    unit = "KB"
                else:
                    memory = nbytes
                    unit = "B"
                return f"{np.result_type(leaf)}{np.shape(leaf)}[size={np.size(leaf)}, bytes={memory:.1f} {unit}]"
            except AttributeError:
                return f"{np.result_type(leaf)}{np.shape(leaf)}[size={np.size(leaf)}]"

        return repr(jax.tree.map(format_size_info, self))

    def save(self, filename: str):
        """
        Serialise the model to file.

        Args:
            filename: the filename
        """
        if not filename.endswith('.pkl'):
            warnings.warn(f"Filename {filename} does not end with .pkl")
        with open(filename, 'wb') as f:
            try:
                pickle.dump(self, f)
            except AttributeError as e:
                if "Can't pickle local object" in str(e):
                    warnings.warn(
                        f"Failed to pickle {self.__class__.__name__}. "
                        f"It's possibly locally defined. Make sure it is globally defined."
                    )
                    raise

    @staticmethod
    def load(filename: str):
        """
        Load the model from file.

        Args:
            filename: the filename

        Returns:
            the model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def build_flatten(cls, this, aux_names: list[str]):
        """Helper function to facilitate dataclass Pytrees."""
        if dataclasses.is_dataclass(this):
            contents = {field.name: getattr(this, field.name) for field in dataclasses.fields(this)}
        else:
            contents = vars(this)

        children_dict = dict(item for item in contents.items() if item[0] not in aux_names)
        aux_data_dict = dict(item for item in contents.items() if item[0] in aux_names)
        return [children_dict], (aux_data_dict,)

    @classmethod
    def build_unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        """Helper function to facilitate dataclass Pytrees."""
        [children_dict] = children
        (aux_data_dict,) = aux_data
        return cls(**children_dict, **aux_data_dict)

    def __reduce__(self):
        children, aux_data = self.flatten(self)
        serialised = (aux_data, children)
        return self._deserialise, (serialised,)

    @classmethod
    def _deserialise(cls, serialised):
        aux_data, children = serialised
        return cls.unflatten(aux_data, children)

    @classmethod
    def register_pytree(cls):
        jax.tree_util.register_pytree_node(cls, cls.flatten, cls.unflatten)

    @classmethod
    @abstractmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        """
        Flatten the model.

        Args:
            this: the model

        Returns:
            the flattened model
        """
        ...

    @classmethod
    @abstractmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        """
        Unflatten the model.

        Args:
            children: the flattened model
            aux_data: the auxiliary

        Returns:
            the unflattened model
        """
        ...

    def to_json(self) -> dict:
        """
        Convert the Pytree to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Pytree.
        """
        children, aux_data = self.flatten(self)

        def _array_to_dict(arr):
            arr_np = np.asarray(arr)
            return {
                'shape': np.shape(arr_np),
                'dtype': str(np.result_type(arr_np)),
                'data': arr_np.tobytes().decode('latin1')
            }

        return {
            'children': jax.tree.map(_array_to_dict, children),
            'aux_data': aux_data,
        }

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        Load the Pytree from a JSON-serializable dictionary.

        Args:
            json_dict: a dictionary representation of the Pytree.
        """

        def _dict_to_array(value):
            return np.frombuffer(
                value['data'].encode('latin1'),
                dtype=value['dtype']
            ).reshape(value['shape'])

        children = jax.tree.map(_dict_to_array, json_dict['children'])
        return cls.unflatten(json_dict['aux_data'], children)


class PureDataclassPytree(Pytree):
    """A Pytree whose dataclass fields are all traceable children."""
    __slots__ = ()

    @classmethod
    def flatten(cls, this) -> tuple[list[Any], tuple[Any, ...]]:
        return cls.build_flatten(this, [])

    @classmethod
    def unflatten(cls, aux_data: tuple[Any, ...], children: list[Any]):
        return cls.build_unflatten(aux_data, children)
