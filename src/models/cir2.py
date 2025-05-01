from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class CIR2(Model):
    def __init__(
        self,
        r0_1: float,
        r0_2: float,
        kappa1: float,
        theta1: float,
        kappa2: float,
        theta2: float,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        if sigma_x is None and sigma_x_center is None:
            raise ValueError("Either sigma_x or sigma_x_center must be provided")
        if sigma_y is None and sigma_y_center is None:
            raise ValueError("Either sigma_y or sigma_y_center must be provided")

        self.r0_1 = r0_1
        self.r0_2 = r0_2
        self.kappa1 = kappa1
        self.theta1 = theta1
        self.kappa2 = kappa2
        self.theta2 = theta2
        self.sigma_x: float = cast(float, sigma_x_center if sigma_x is None else sigma_x)
        self.sigma_y: float = cast(float, sigma_y_center if sigma_y is None else sigma_y)
        self._sigma_x_bounds: Optional[Tuple[float, float]] = None
        self._sigma_y_bounds: Optional[Tuple[float, float]] = None

        if sigma_x_center is not None:
            self._sigma_x_bounds = (sigma_x_center * 0.95, sigma_x_center * 1.05)

        if sigma_y_center is not None:
            self._sigma_y_bounds = (sigma_y_center * 0.95, sigma_y_center * 1.05)

    def __str__(self) -> str:
        return (
            f"--- CIR2 ---\n"
            f"r0_1={self.r0_1}\n"
            f"r0_2={self.r0_2}\n"
            f"kappa1={self.kappa1}\n"
            f"theta1={self.theta1}\n"
            f"sigma_x={self.sigma_x}\n"
            f"kappa2={self.kappa2}\n"
            f"theta2={self.theta2}\n"
            f"sigma_y={self.sigma_y}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.r0_1,
                self.r0_2,
                self.kappa1,
                self.theta1,
                self.kappa2,
                self.theta2,
                self.sigma_x,
                self.sigma_y,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "CIR2":  # type: ignore
        return cls(
            r0_1=float(arr[0]),
            r0_2=float(arr[1]),
            kappa1=float(arr[2]),
            theta1=float(arr[3]),
            kappa2=float(arr[4]),
            theta2=float(arr[5]),
            sigma_x=float(arr[6]),
            sigma_y=float(arr[7]),
        )

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.001, -np.inf, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma_x_bounds is not None:
            lower[4] = self._sigma_x_bounds[0]
            upper[4] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[7] = self._sigma_y_bounds[0]
            upper[7] = self._sigma_y_bounds[1]

        return lower, upper

    def params(self) -> "CIR2":
        return self

    def update(self, p: "Model") -> None:
        assert isinstance(p, CIR2)
        self.r0_1 = p.r0_1
        self.r0_2 = p.r0_2
        self.kappa1 = p.kappa1
        self.theta1 = p.theta1
        self.sigma_x = p.sigma_x
        self.kappa2 = p.kappa2
        self.theta2 = p.theta2
        self.sigma_y = p.sigma_y
