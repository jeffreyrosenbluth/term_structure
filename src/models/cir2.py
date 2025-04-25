from typing import Tuple, TypeVar

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
        sigma1: float,
        kappa2: float,
        theta2: float,
        sigma2: float,
    ) -> None:
        self.r0_1 = r0_1
        self.r0_2 = r0_2
        self.kappa1 = kappa1
        self.theta1 = theta1
        self.sigma1 = sigma1
        self.kappa2 = kappa2
        self.theta2 = theta2
        self.sigma2 = sigma2

    # numerical helpers for calibrator / optimizer
    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.r0_1,
                self.r0_2,
                self.kappa1,
                self.theta1,
                self.sigma1,
                self.kappa2,
                self.theta2,
                self.sigma2,
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
            f"sigma1={self._params.sigma1}\n"
            f"kappa2={self._params.kappa2}\n"
            f"theta2={self._params.theta2}\n"
            f"sigma2={self._params.sigma2}\n"
        )

    def params(self) -> CIR2Params:
        return self._params

    def update_params(self, p: CIR2Params) -> None:
        self._params = p
