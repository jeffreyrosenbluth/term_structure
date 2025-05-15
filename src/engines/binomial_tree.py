"""Binomial tree pricing engine for interest rate models.

This module implements binomial tree pricing methods for various interest rate models.
The binomial tree approach discretizes the continuous-time stochastic processes into
a discrete-time lattice, allowing for numerical pricing of zero-coupon bonds.

The module currently supports:
- Vasicek model: Uses a mean-reverting binomial tree with state-dependent probabilities
- Merton model: Uses a simple binomial tree with constant probabilities

Each model's implementation follows the specific characteristics of its underlying
stochastic process while maintaining the general binomial tree framework.
"""

import numpy as np

from src.core.binomial import BinomialTree
from src.models.merton import Merton
from src.models.vasicek import Vasicek


def price_binomial_vasicek(model: Vasicek, T: float, maxT: float, dt: float) -> float:
    """Calculate zero-coupon bond price using a binomial tree for the Vasicek model.

    The Vasicek model's binomial tree implementation uses state-dependent probabilities
    to capture the mean-reverting nature of the process. The tree is constructed with:
    - Up/down movements: ±σ√dt
    - State-dependent probabilities: 0.5 + κ(θ-r)√dt/(2σ)
    where κ is the mean reversion speed, θ is the long-term mean, and σ is volatility.

    Args:
        model: A Vasicek model instance containing the model parameters
        T: Time to maturity in years
        maxT: Maximum time horizon for the tree construction
        dt: Time step size for the tree discretization

    Returns:
        float: The zero-coupon bond price

    Note:
        The tree construction uses a state-dependent probability to maintain the
        mean-reverting property of the Vasicek process. The probability is clipped
        to [0,1] to ensure numerical stability.
    """
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
    """Calculate zero-coupon bond price using a binomial tree for the Merton model.

    The Merton model's binomial tree implementation uses constant probabilities and
    drift-adjusted movements to capture the Brownian motion with drift. The tree is
    constructed with:
    - Up/down movements: μdt ± σ√dt
    - Constant probabilities: 0.5
    where μ is the drift parameter and σ is volatility.

    Args:
        model: A Merton model instance containing the model parameters
        T: Time to maturity in years
        maxT: Maximum time horizon for the tree construction
        dt: Time step size for the tree discretization

    Returns:
        float: The zero-coupon bond price

    Note:
        The tree construction uses constant probabilities of 0.5 since the Merton
        model is a simple Brownian motion with drift, without mean reversion.
    """
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
