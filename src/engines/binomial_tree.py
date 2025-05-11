import numpy as np

from src.core.binomial import BinomialTree
from src.models.merton import Merton
from src.models.vasicek import Vasicek


def price_binomial_vasicek(model: Vasicek, T: float, maxT: float, dt: float) -> float:
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma

    def probability(r: float) -> float:
        return float(0.5 + kappa * (theta - r) * np.sqrt(dt) / (2.0 * sigma))

    periods = 1 + int(maxT / dt)
    node_count = int((periods + 1) * (periods + 2) / 2)
    tree = np.empty(node_count)
    tree[0] = r0
    up_probs = np.empty(node_count)

    # Binomial Tree
    for i in range(0, int(periods * (periods + 1) / 2)):
        (d, u) = BinomialTree.children(i)
        tree[d] = tree[i] - sigma * np.sqrt(dt)
        tree[u] = tree[i] + sigma * np.sqrt(dt)
        up_probs[i] = np.clip(probability(tree[i]), 0, 1)

    bTree = BinomialTree(tree, up_probs, periods, dt, node_count)
    return bTree.price(T)


def price_binomial_merton(model: Merton, T: float, maxT: float, dt: float) -> float:
    r0, mu, sigma = model.r0, model.mu, model.sigma
    periods = 1 + int(maxT / dt)
    node_count = int((periods + 1) * (periods + 2) / 2)
    tree = np.empty(node_count)
    tree[0] = r0
    up_probs = np.ones(node_count) * 0.5

    # Binomial Tree
    for i in range(0, int(periods * (periods + 1) / 2)):
        (d, u) = BinomialTree.children(i)
        tree[d] = tree[i] + mu * dt - sigma * np.sqrt(dt)
        tree[u] = tree[i] + mu * dt + sigma * np.sqrt(dt)

    bTree = BinomialTree(tree, up_probs, periods, dt, node_count)
    return bTree.price(T)
