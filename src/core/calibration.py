from typing import Any, Generic, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.core.engine import PricingEngine
from src.core.model import Model

P = TypeVar("P", bound=Model)


class Calibrator(Generic[P]):
    def __init__(
        self,
        model: P,
        engine: PricingEngine[P],
        solver: Any,
    ) -> None:
        self.model = model
        self.engine = engine
        self.solver = solver

    def calibrate(self, market: Sequence[Tuple[float, float]]) -> None:
        t, y = map(np.asarray, zip(*market))

        def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            params_cls = type(self.model.params())
            self.model.update_params(params_cls.from_array(theta))
            prices = np.array([self.engine.P(self.model, ti) for ti in t])
            return self.engine.spot_rate(prices, t) - y

        lower, upper = self.model.params().get_bounds()

        theta0 = self.model.params().to_array()
        theta_star = self.solver.minimize(
            theta0,
            residuals,
            bounds=(lower, upper),
        )
        params_cls = type(self.model.params())
        self.model.update_params(params_cls.from_array(theta_star))
