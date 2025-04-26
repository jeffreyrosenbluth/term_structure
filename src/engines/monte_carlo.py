import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.engine import PricingEngine
from src.core.model import ShortRateModel
from src.models.cir import CIRParams
from src.models.cir2 import CIR2Params
from src.models.g2 import G2Params
from src.models.merton import MertonParams
from src.models.vasicek import VasicekParams


class MonteCarloMerton(PricingEngine[MertonParams]):
    def __init__(self, maxT: float, dt: float, num_paths: int = 10_000) -> None:
        self.maxT = maxT
        self.dt = dt
        self.num_paths = num_paths

    def P(self, model: ShortRateModel[MertonParams], T: float) -> float:
        np.random.seed(13131313)
        p = model.params()
        r0, mu, sigma = p.r0, p.mu, p.sigma

        sdt = np.sqrt(self.dt)
        paths = mu * self.dt + sigma * sdt * np.random.randn(
            self.num_paths, int(self.maxT / self.dt)
        )
        paths[:, 0] = r0
        paths = np.cumsum(paths, axis=1)
        return float(np.mean(np.exp(-np.sum(self.dt * paths[:, : int(T / self.dt)], axis=1))))


class MonteCarloVasicek(PricingEngine[VasicekParams]):
    def __init__(self, maxT: float, dt: float, num_paths: int = 10_000) -> None:
        self.maxT = maxT
        self.dt = dt
        self.num_paths = num_paths

    def P(self, model: ShortRateModel[VasicekParams], T: float) -> float:
        np.random.seed(13131313)
        p = model.params()
        r0, kappa, theta, sigma = p.r0, p.kappa, p.theta, p.sigma
        num_steps = int(self.maxT / self.dt)
        paths = np.empty([self.num_paths, 1 + num_steps])
        paths[:, 0] = r0
        dW = np.random.randn(self.num_paths, num_steps)

        for t in range(num_steps):
            paths[:, t + 1] = (
                paths[:, t]
                + kappa * (theta - paths[:, t]) * self.dt
                + np.sqrt(self.dt) * sigma * dW[:, t]
            )
        return float(np.mean(np.exp(-np.sum(self.dt * paths[:, : int(T / self.dt)], axis=1))))
