import math
from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model

P = TypeVar("P", bound=Model)


class PricingEngine(ABC, Generic[P]):
    # ----- user must implement this ----------------------------------
    @abstractmethod
    def P(self, model: P, T: float) -> float: ...

    # -----------------------------------------------------------------
    @staticmethod
    def spot_rate(
        price: Union[float, Sequence[float], NDArray[np.float64]],
        T: Union[float, Sequence[float], NDArray[np.float64]],
    ) -> Union[float, NDArray[np.float64]]:
        price_arr = np.asarray(price, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        if np.any(T_arr == 0.0):
            raise ValueError("T cannot be (or contain) zero")
        with np.errstate(divide="ignore", invalid="ignore"):
            y = -np.log(price_arr) / T_arr
        return np.where(np.isfinite(y), y, np.inf)  # or some large cap


def R(engine: PricingEngine[P], model: P, T: float) -> float:
    if T == 0.0:
        raise ValueError("T cannot be zero in R()")
    return -math.log(engine.P(model, T)) / T
