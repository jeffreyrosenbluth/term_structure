from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class GV2P(Model):
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

    def __str__(self) -> str:
        return (
            f"--- GV2P ---\n"
            f"x0={self.x0}\n"
            f"y0={self.y0}\n"
            f"z0={self.z0}\n"
            f"lambda_={self.lambda_}\n"
            f"gamma={self.gamma}\n"
            f"sigma_x={self.sigma_x}\n"
            f"sigma_y={self.sigma_y}\n"
            f"k={self.k}\n"
            f"phi={self.phi}\n"
        )

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
    def from_array(cls, a: NDArray[np.float64]) -> "GV2P":
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

    def params(self) -> "GV2P":
        return self

    def update_params(self, p: "Model") -> None:
        assert isinstance(p, GV2P)
        self.x0 = p.x0
        self.y0 = p.y0
        self.z0 = p.z0
        self.lambda_ = p.lambda_
        self.gamma = p.gamma
        self.sigma_x = p.sigma_x
        self.sigma_y = p.sigma_y
        self.k = p.k
        self.phi = p.phi
