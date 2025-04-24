from typing import Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

VP = TypeVar("VP", bound="VasicekParams")


class VasicekParams(Parameters):
    # Fixed parameter
    sigma: float = 0.01

    def __init__(
        self, r0: float, kappa: float, theta: float, sigma: Optional[float] = None
    ) -> None:
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        if sigma is not None:
            self.sigma = sigma  # Update class variable if provided

    def to_array(self) -> NDArray[np.float64]:
        # Only include parameters to be calibrated
        return np.array([self.r0, self.kappa, self.theta], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "VasicekParams":  # type: ignore
        # Keep sigma fixed, only update other parameters
        return cls(arr[0], arr[1], arr[2])


class Vasicek(ShortRateModel[VasicekParams]):
    def __init__(self, params: VasicekParams) -> None:
        self._params = params

    def params(self) -> VasicekParams:
        return self._params

    def update_params(self, p: VasicekParams) -> None:
        self._params = p
