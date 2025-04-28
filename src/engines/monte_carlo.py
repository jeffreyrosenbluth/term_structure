import math

import numpy as np

from src.core.engine import PricingEngine
from src.core.model import Model
from src.models.merton import Merton
from src.models.vasicek import Vasicek


class MonteCarloMerton(PricingEngine[Merton]):
    def __init__(self, maxT: float, dt: float, num_paths: int = 10_000) -> None:
        self.maxT = maxT
        self.dt = dt
        self.num_paths = num_paths
        self._initialize_paths()

    def _initialize_paths(self) -> None:
        num_steps = int(self.maxT / self.dt)
        self._dW = np.random.randn(self.num_paths, num_steps)
        self._cumsum_dW = np.cumsum(self._dW, axis=1)
        self._time_steps = np.arange(1, num_steps + 1) * self.dt

    def P(self, model: Model, T: float) -> float:
        assert isinstance(model, Merton)
        p = model.params()
        r0, mu, sigma = p.r0, p.mu, p.sigma
        steps = int(T / self.dt)
        if steps == 0:
            return math.exp(-r0 * T)
        paths = (
            r0
            + mu * self._time_steps[:steps]
            + sigma * np.sqrt(self.dt) * self._cumsum_dW[:, :steps]
        )
        discount_factors = np.exp(-np.sum(self.dt * paths, axis=1))
        return float(np.mean(discount_factors))


class MonteCarloVasicek(PricingEngine[Vasicek]):
    def __init__(self, maxT: float, dt: float, num_paths: int = 10_000) -> None:
        self.maxT = maxT
        self.dt = dt
        self.num_paths = num_paths
        self._initialize_paths()

    def _initialize_paths(self) -> None:
        num_steps = int(self.maxT / self.dt)
        self._dW = np.random.randn(self.num_paths, num_steps)
        self._time_steps = np.arange(1, num_steps + 1) * self.dt

    def P(self, model: Model, T: float) -> float:
        assert isinstance(model, Vasicek)
        p = model.params()
        r0, kappa, theta, sigma = p.r0, p.kappa, p.theta, p.sigma
        steps = int(T / self.dt)
        if steps == 0:
            return math.exp(-r0 * T)

        paths = np.empty((self.num_paths, steps + 1))
        paths[:, 0] = r0
        for t in range(steps):
            paths[:, t + 1] = (
                paths[:, t]
                + kappa * (theta - paths[:, t]) * self.dt
                + np.sqrt(self.dt) * sigma * self._dW[:, t]
            )

        discount_factors = np.exp(-np.sum(self.dt * paths[:, :steps], axis=1))
        return float(np.mean(discount_factors))
