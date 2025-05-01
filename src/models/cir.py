from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class CIR(Model):
    def __init__(
        self,
        r0: float,
        kappa: float,
        theta: float,
        sigma: Optional[float] = None,
        sigma_center: Optional[float] = None,
    ) -> None:
        if sigma is None and sigma_center is None:
            raise ValueError("Either sigma or sigma_center must be provided")

        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma_center if sigma is None else sigma
        self._sigma_bounds: Optional[Tuple[float, float]] = None

        if sigma_center is not None:
            self._sigma_bounds = (sigma_center * 0.95, sigma_center * 1.05)

    def __str__(self) -> str:
        return (
            f"--- CIR ---\n"
            f"r0={self.r0}\n"
            f"kappa={self.kappa}\n"
            f"theta={self.theta}\n"
            f"sigma={self.sigma}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.r0, self.kappa, self.theta, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "CIR":  # type: ignore
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([0.000, 0.000, 0.000, 0.001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma_bounds is not None:
            lower[3] = self._sigma_bounds[0]
            upper[3] = self._sigma_bounds[1]

        return lower, upper

    def params(self) -> "Model":
        return self

    def update_params(self, p: "Model") -> None:
        assert isinstance(p, CIR)
        self.r0 = p.r0
        self.kappa = p.kappa
        self.theta = p.theta
        self.sigma = p.sigma
