from __future__ import annotations  # enables typing.Self on 3.11+

from typing import Protocol, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

P_co = TypeVar("P_co", bound="Parameters", covariant=True)


class Parameters(Protocol):
    def to_array(self) -> NDArray[np.float64]: ...

    @classmethod
    def from_array(cls: type[P_co], a: NDArray[np.float64]) -> P_co: ...

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
