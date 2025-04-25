from typing import Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

VP = TypeVar("VP", bound="VasicekParams")


class VasicekParams(Parameters):
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float) -> None:
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def to_array(self) -> NDArray[np.float64]:
        # Only include parameters to be calibrated
        return np.array([self.r0, self.kappa, self.theta, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "VasicekParams":  # type: ignore
        # Keep sigma fixed, only update other parameters
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, 0.001, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf])
        return lower, upper


class Vasicek(ShortRateModel[VasicekParams]):
    def __init__(self, params: VasicekParams) -> None:
        self._params = params

    def __str__(self) -> str:
        return (
            f"--- Vasicek ---\n"
            f"r0={self._params.r0}\n"
            f"kappa={self._params.kappa}\n"
            f"theta={self._params.theta}\n"
            f"sigma={self._params.sigma}\n"
        )

    def params(self) -> VasicekParams:
        return self._params

    def update_params(self, p: VasicekParams) -> None:
        self._params = p
