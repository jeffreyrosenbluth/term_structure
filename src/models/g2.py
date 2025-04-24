from typing import Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

G2P = TypeVar("G2P", bound="G2Params")


class G2Params(Parameters):
    sigma_x: float = 0.05
    sigma_y: float = 0.05

    def __init__(
        self,
        r0: float,
        a: float,
        b: float,
        rho: float,
        phi: float,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
    ) -> None:
        self.r0 = r0
        self.a = a
        self.b = b
        self.rho = rho
        self.phi = phi
        if sigma_x is not None:
            self.sigma_x = sigma_x
        if sigma_y is not None:
            self.sigma_y = sigma_y

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.r0, self.a, self.b, self.rho, self.phi], dtype=np.float64)

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "G2Params":  # type: ignore
        return cls(a[0], a[1], a[2], a[3], a[4])


class G2(ShortRateModel[G2Params]):
    def __init__(self, params: G2Params) -> None:
        self._params = params

    def params(self) -> G2Params:
        return self._params

    def update_params(self, p: G2Params) -> None:
        self._params = p
