from typing import Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

GV = TypeVar("GV", bound="GV2PParams")


class GV2PParams(Parameters):
    def __init__(
        self,
        x0: float,
        y0: float,
        z0: float,
        lambda_: float,
        gamma: float,
        sigma_x: float,
        sigma_y: float,
        k: float,
        phi: float,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.lambda_ = lambda_
        self.gamma = gamma
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.k = k
        self.phi = phi

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.x0,
                self.y0,
                self.z0,
                self.lambda_,
                self.gamma,
                self.sigma_x,
                self.sigma_y,
                self.k,
                self.phi,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "GV2PParams":
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                0.0,
                0.0,
                0.0001,
                0.0001,
                0.0001,
                -np.inf,
            ]
        )
        upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        return lower, upper


class GV2P(ShortRateModel[GV2PParams]):
    def __init__(self, params: GV2PParams) -> None:
        self._params = params

    def __str__(self) -> str:
        return (
            f"--- GV2P ---\n"
            f"x0={self._params.x0}\n"
            f"y0={self._params.y0}\n"
            f"z0={self._params.z0}\n"
            f"lambda_={self._params.lambda_}\n"
            f"gamma={self._params.gamma}\n"
            f"sigma_x={self._params.sigma_x}\n"
            f"sigma_y={self._params.sigma_y}\n"
            f"k={self._params.k}\n"
            f"phi={self._params.phi}\n"
        )

    def params(self) -> GV2PParams:
        return self._params

    def update_params(self, p: GV2PParams) -> None:
        self._params = p
