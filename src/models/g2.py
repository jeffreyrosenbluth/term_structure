from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class G2(Model):
    def __init__(
        self,
        x0: float,
        y0: float,
        a: float,
        b: float,
        rho: float,
        phi: float,
        sigma_x: float,
        sigma_y: float,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b
        self.rho = rho
        self.phi = phi
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self._sigma_x_bounds: Optional[Tuple[float, float]] = None
        self._sigma_y_bounds: Optional[Tuple[float, float]] = None

        if sigma_x_center is not None:
            self._sigma_x_bounds = (sigma_x_center * 0.95, sigma_x_center * 1.05)

        if sigma_y_center is not None:
            self._sigma_y_bounds = (sigma_y_center * 0.95, sigma_y_center * 1.05)

    def __str__(self) -> str:
        return (
            f"--- G2 ---\n"
            f"x0={self.x0}\n"
            f"y0={self.y0}\n"
            f"a={self.a}\n"
            f"b={self.b}\n"
            f"rho={self.rho}\n"
            f"phi={self.phi}\n"
            f"sigma_x={self.sigma_x}\n"
            f"sigma_y={self.sigma_y}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [self.x0, self.y0, self.a, self.b, self.rho, self.phi, self.sigma_x, self.sigma_y],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "G2":  # type: ignore
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -1.0, -np.inf, 0.0001, 0.0001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma_x_bounds is not None:
            lower[6] = self._sigma_x_bounds[0]
            upper[6] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[7] = self._sigma_y_bounds[0]
            upper[7] = self._sigma_y_bounds[1]

        return lower, upper

    def params(self) -> "G2":
        return self

    def update_params(self, p: "Model") -> None:
        assert isinstance(p, G2)
        self.x0 = p.x0
        self.y0 = p.y0
        self.a = p.a
        self.b = p.b
        self.rho = p.rho
        self.phi = p.phi
        self.sigma_x = p.sigma_x
        self.sigma_y = p.sigma_y
