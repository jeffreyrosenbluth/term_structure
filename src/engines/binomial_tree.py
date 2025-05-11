import numpy as np

from src.core.binomial import BinomialTree
from src.core.engine import PricingEngine
from src.core.model import Model
from src.models.merton import Merton
from src.models.vasicek import Vasicek


class BinomialVasicek(PricingEngine[Vasicek]):
    def __init__(self, maxT: float, dt: float) -> None:
        self.maxT = maxT
        self.dt = dt

    def P(self, model: Model, T: float) -> float:
        assert isinstance(model, Vasicek)
        r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma

        def probability(r: float) -> float:
            return float(0.5 + kappa * (theta - r) * np.sqrt(self.dt) / (2.0 * sigma))

        periods = 1 + int(self.maxT / self.dt)
        node_count = int((periods + 1) * (periods + 2) / 2)
        tree = np.empty(node_count)
        tree[0] = r0
        up_probs = np.empty(node_count)

        # Binomial Tree
        for i in range(0, int(periods * (periods + 1) / 2)):
            (d, u) = BinomialTree.children(i)
            tree[d] = tree[i] - sigma * np.sqrt(self.dt)
            tree[u] = tree[i] + sigma * np.sqrt(self.dt)
            up_probs[i] = np.clip(probability(tree[i]), 0, 1)

        self.bTree = BinomialTree(tree, up_probs, periods, self.dt, node_count)
        return self.bTree.price(T)


class BinomialMerton(PricingEngine[Merton]):
    def __init__(self, maxT: float, dt: float) -> None:
        self.maxT = maxT
        self.dt = dt

    def P(self, model: Model, T: float) -> float:
        assert isinstance(model, Merton)
        r0, mu, sigma = model.r0, model.mu, model.sigma
        periods = 1 + int(self.maxT / self.dt)
        node_count = int((periods + 1) * (periods + 2) / 2)
        tree = np.empty(node_count)
        tree[0] = r0
        up_probs = np.ones(node_count) * 0.5

        # Binomial Tree
        for i in range(0, int(periods * (periods + 1) / 2)):
            (d, u) = BinomialTree.children(i)
            tree[d] = tree[i] + mu * self.dt - sigma * np.sqrt(self.dt)
            tree[u] = tree[i] + mu * self.dt + sigma * np.sqrt(self.dt)

        self.bTree = BinomialTree(tree, up_probs, periods, self.dt, node_count)
        return self.bTree.price(T)
