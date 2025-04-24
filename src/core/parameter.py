from typing import Protocol, TypeVar

import numpy as np

P = TypeVar("P", bound="Parameters")


class Parameters(Protocol):
    """Fixed-size parameter block that exposes a NumPy 1-D view."""

    def to_array(self) -> np.ndarray: ...
    @classmethod
    def from_array(cls: type[P], a: np.ndarray) -> P: ...
