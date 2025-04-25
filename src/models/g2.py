from typing import Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

G2P = TypeVar("G2P", bound="G2Params")


class G2Params(Parameters):
    def __init__(
        self,
        a: float,
        b: float,
        rho: float,
        phi: float,
        sigma_x: float,
        sigma_y: float,
    ) -> None:
        self.a = a
        self.b = b
        self.rho = rho
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [self.a, self.b, self.rho, self.phi, self.sigma_x, self.sigma_y],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "G2Params":  # type: ignore
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, -np.inf, -1.0, -np.inf, 0.0001, 0.0001])
        upper = np.array([np.inf, np.inf, 1.0, np.inf, 1.0, 1.0])
        return lower, upper


class G2(ShortRateModel[G2Params]):
    def __init__(self, params: G2Params) -> None:
        self._params = params

    def __str__(self) -> str:
        return (
            f"--- G2 ---\n"
            f"a={self._params.a}\n"
            f"b={self._params.b}\n"
            f"rho={self._params.rho}\n"
            f"phi={self._params.phi}\n"
            f"sigma_x={self._params.sigma_x}\n"
            f"sigma_y={self._params.sigma_y}\n"
        )

    def params(self) -> G2Params:
        return self._params

    def update_params(self, p: G2Params) -> None:
        self._params = p
