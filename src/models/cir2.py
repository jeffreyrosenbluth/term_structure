from typing import Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

ParamT = TypeVar("ParamT", bound=Parameters)


class CIR2Params(Parameters):
    def __init__(
        self,
        r0_1: float,
        r0_2: float,
        kappa1: float,
        theta1: float,
        sigma_x: float,
        kappa2: float,
        theta2: float,
        sigma_y: float,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        self.r0_1 = r0_1
        self.r0_2 = r0_2
        self.kappa1 = kappa1
        self.theta1 = theta1
        self.sigma_x = sigma_x
        self.kappa2 = kappa2
        self.theta2 = theta2
        self.sigma_y = sigma_y
        self.sigma_x_center = sigma_x_center
        self.sigma_y_center = sigma_y_center

        self._sigma_x_bounds: Optional[Tuple[float, float]] = None
        self._sigma_y_bounds: Optional[Tuple[float, float]] = None

        if sigma_x_center is not None:
            self._sigma_x_bounds = (sigma_x_center * 0.95, sigma_x_center * 1.05)

        if sigma_y_center is not None:
            self._sigma_y_bounds = (sigma_y_center * 0.95, sigma_y_center * 1.05)

    # numerical helpers for calibrator / optimizer
    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.r0_1,
                self.r0_2,
                self.kappa1,
                self.theta1,
                self.sigma_x,
                self.kappa2,
                self.theta2,
                self.sigma_y,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "CIR2Params":
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lo = np.array(
            [-np.inf, -np.inf, 0.0, 0.0, 0.001, 0.0, 0.0, 0.001],
            dtype=np.float64,
        )
        hi = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            dtype=np.float64,
        )
        return lo, hi

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self.sigma_x_center is not None:
            lower[4] = self.sigma_x_center * 0.95
            upper[4] = self.sigma_x_center * 1.05

        if self.sigma_y_center is not None:
            lower[7] = self.sigma_y_center * 0.95
            upper[7] = self.sigma_y_center * 1.05

        return lower, upper


class CIR2(ShortRateModel[CIR2Params]):
    def __init__(self, params: CIR2Params) -> None:
        self._params = params

    def __str__(self) -> str:
        return (
            f"--- CIR2 ---\n"
            f"r0_1={self._params.r0_1}\n"
            f"r0_2={self._params.r0_2}\n"
            f"kappa1={self._params.kappa1}\n"
            f"theta1={self._params.theta1}\n"
            f"sigma_x={self._params.sigma_x}\n"
            f"kappa2={self._params.kappa2}\n"
            f"theta2={self._params.theta2}\n"
            f"sigma_y={self._params.sigma_y}\n"
        )

    def params(self) -> CIR2Params:
        return self._params

    def update_params(self, p: CIR2Params) -> None:
        self._params = p
