from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class V2(Model):
    def __init__(
        self,
        y1_0: float,
        y2_0: float,
        k11: float,
        k21: float,
        k22: float,
        delta0: float,
        delta1: float,
        delta2: float,
        sigma1: float,
        sigma2: float,
        sigma1_center: Optional[float] = None,
        sigma2_center: Optional[float] = None,
    ) -> None:
        self.y1_0 = y1_0
        self.y2_0 = y2_0
        self.k11 = k11
        self.k21 = k21
        self.k22 = k22
        self.delta0 = delta0
        self.delta1 = delta1
        self.delta2 = delta2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self._sigma1_bounds: Optional[Tuple[float, float]] = None
        self._sigma2_bounds: Optional[Tuple[float, float]] = None

        if sigma1_center is not None:
            self._sigma1_bounds = (sigma1_center * 0.95, sigma1_center * 1.05)

        if sigma2_center is not None:
            self._sigma2_bounds = (sigma2_center * 0.95, sigma2_center * 1.05)

    def __str__(self) -> str:
        return (
            f"--- V2 ---\n"
            f"y1_0={self.y1_0}\n"
            f"y2_0={self.y2_0}\n"
            f"k11={self.k11}\n"
            f"k21={self.k21}\n"
            f"k22={self.k22}\n"
            f"delta0={self.delta0}\n"
            f"delta1={self.delta1}\n"
            f"delta2={self.delta2}\n"
            f"sigma1={self.sigma1}\n"
            f"sigma2={self.sigma2}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.y1_0,
                self.y2_0,
                self.k11,
                self.k21,
                self.k22,
                self.delta0,
                self.delta1,
                self.delta2,
                self.sigma1,
                self.sigma2,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "V2":
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array(
            [-np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf, -np.inf, -np.inf, 0.001, 0.001]
        )
        upper = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        )
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower, upper = self.bounds()

        if self._sigma1_bounds is not None:
            lower[8] = self._sigma1_bounds[0]
            upper[8] = self._sigma1_bounds[1]

        if self._sigma2_bounds is not None:
            lower[9] = self._sigma2_bounds[0]
            upper[9] = self._sigma2_bounds[1]

        return lower, upper

    def params(self) -> "V2":
        return self

    def update_params(self, p: "Model") -> None:
        assert isinstance(p, V2)
        self.y1_0 = p.y1_0
        self.y2_0 = p.y2_0
        self.k11 = p.k11
        self.k21 = p.k21
        self.k22 = p.k22
        self.delta0 = p.delta0
        self.delta1 = p.delta1
        self.delta2 = p.delta2
        self.sigma1 = p.sigma1
        self.sigma2 = p.sigma2
