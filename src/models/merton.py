from typing import Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

MP = TypeVar("MP", bound="MertonParams")


class MertonParams(Parameters):
    def __init__(self, r0: float, mu: float, sigma: float) -> None:
        self.r0 = r0
        self.mu = mu
        self.sigma = sigma

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.r0, self.mu, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "MertonParams":  # type: ignore
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf])
        return lower, upper


class Merton(ShortRateModel[MertonParams]):
    def __init__(self, params: MertonParams) -> None:
        self._params = params

    def __str__(self) -> str:
        return (
            f"--- Merton ---\n"
            f"r0={self._params.r0}\n"
            f"mu={self._params.mu}\n"
            f"sigma={self._params.sigma}\n"
        )

    def params(self) -> MertonParams:
        return self._params

    def update_params(self, p: MertonParams) -> None:
        self._params = p
