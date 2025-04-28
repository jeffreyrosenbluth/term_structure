from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class Merton(Model):
    def __init__(
        self, r0: float, mu: float, sigma: float, sigma_center: Optional[float] = None
    ) -> None:
        self.r0 = r0
        self.mu = mu
        self.sigma = sigma
        self._sigma_bounds: Optional[Tuple[float, float]] = None

        if sigma_center is not None:
            self._sigma_bounds = (sigma_center * 0.95, sigma_center * 1.05)

    def __str__(self) -> str:
        return f"--- Merton ---\nr0={self.r0}\nmu={self.mu}\nsigma={self.sigma}\n"

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.r0, self.mu, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Merton":  # type: ignore
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma_bounds is not None:
            lower[2] = self._sigma_bounds[0]
            upper[2] = self._sigma_bounds[1]

        return lower, upper

    def params(self) -> "Model":
        return self

    def update_params(self, p: "Model") -> None:
        assert isinstance(p, Merton)
        self.r0 = p.r0
        self.mu = p.mu
        self.sigma = p.sigma
