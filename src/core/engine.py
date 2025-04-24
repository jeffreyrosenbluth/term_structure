# core/engine.py
import math
from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from src.core.model import ShortRateModel
from src.core.parameter import Parameters

PT = TypeVar("PT", bound=Parameters)


class PricingEngine(ABC, Generic[PT]):
    @abstractmethod
    def P(self, model: ShortRateModel[PT], T: float) -> float: ...

    def R(self, model: ShortRateModel[PT], T: float) -> float:
        return -math.log(self.P(model, T)) / T

    @staticmethod
    def spot_rate(
        price: Union[float, Sequence[float]], T: Union[float, Sequence[float]]
    ) -> Union[float, NDArray[np.float64]]:
        if isinstance(T, (list, np.ndarray)) and any(t == 0 for t in T):
            raise ValueError("T cannot contain zero")
        if isinstance(T, float) and T == 0:
            raise ValueError("T cannot be zero")
        return -np.log(price) / T
