from __future__ import annotations  # enables typing.Self on 3.11+

from typing import Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class Model(Protocol):
    def to_array(self) -> NDArray[np.float64]: ...

    @classmethod
    def from_array(cls: type["Model"], a: NDArray[np.float64]) -> "Model": ...

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def update(self, p: "Model") -> None: ...
