from __future__ import annotations  # enables typing.Self on 3.11+

from typing import Protocol, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

P = TypeVar("P", bound="Model")


class Model(Protocol):
    def to_array(self) -> NDArray[np.float64]: ...

    @classmethod
    def from_array(cls: type[P], a: NDArray[np.float64]) -> P: ...

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def params(self: P) -> P: ...

    def update_params(self, p: P) -> None: ...
