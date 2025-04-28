from typing import Optional, Tuple, TypeVar

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
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
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

        # Calculate bounds if center values are provided
        self._sigma_x_bounds: Optional[Tuple[float, float]] = None
        self._sigma_y_bounds: Optional[Tuple[float, float]] = None

        if sigma_x_center is not None:
            self._sigma_x_bounds = (sigma_x_center * 0.95, sigma_x_center * 1.05)

        if sigma_y_center is not None:
            self._sigma_y_bounds = (sigma_y_center * 0.95, sigma_y_center * 1.05)

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

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma_x_bounds is not None:
            lower[5] = self._sigma_x_bounds[0]
            upper[5] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[6] = self._sigma_y_bounds[0]
            upper[6] = self._sigma_y_bounds[1]

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
