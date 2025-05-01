from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class Merton(Model):
    def __init__(
        self,
        r0: float,
        mu: float,
        sigma: Optional[float] = None,
        sigma_center: Optional[float] = None,
    ) -> None:
        if sigma is None and sigma_center is None:
            raise ValueError("Either sigma or sigma_center must be provided")

        self.r0 = r0
        self.mu = mu
        self.sigma: float = cast(float, sigma_center if sigma is None else sigma)
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

    def update(self, p: "Model") -> None:
        assert isinstance(p, Merton)
        self.r0 = p.r0
        self.mu = p.mu
        self.sigma = p.sigma
