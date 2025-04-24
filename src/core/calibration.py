from typing import Any, Generic, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.engine import PricingEngine
from src.core.model import ShortRateModel
from src.core.parameter import Parameters

P = TypeVar("P", bound=Parameters)


class Calibrator(Generic[P]):
    def __init__(self, model: ShortRateModel[P], engine: PricingEngine[P], solver: Any) -> None:
        self.model = model
        self.engine = engine
        self.solver = solver

    def calibrate(self, market: Sequence[Tuple[float, float]]) -> None:
        t, y = map(np.asarray, zip(*market))

        def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            self.model.update_params(self.model.params().from_array(theta))
            prices = np.array([self.engine.P(self.model, ti) for ti in t])
            return self.engine.spot_rate(prices, t) - y

        theta_star = self.solver.minimize(self.model.params().to_array(), residuals)
        self.model.update_params(self.model.params().from_array(theta_star))
