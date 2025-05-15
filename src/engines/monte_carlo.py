"""Monte Carlo simulation engine for interest rate models.

This module implements Monte Carlo simulation methods for pricing zero-coupon bonds
under various interest rate models. The Monte Carlo approach simulates multiple
paths of the short rate process and computes bond prices by averaging the discounted
payoffs across all paths.

The module currently supports:
- Merton model: Brownian motion with drift
- Vasicek model: Mean-reverting Ornstein-Uhlenbeck process
- G2 model: Two-factor Gaussian model

Each model's implementation includes both pricing and path visualization capabilities.
The simulations use Euler-Maruyama discretization for the stochastic differential
equations governing the interest rate processes.
"""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy.typing import NDArray

from src.models.g2 import G2
from src.models.merton import Merton
from src.models.vasicek import Vasicek


def plot_paths(paths: NDArray[np.float64], dt: float, n: int) -> Figure:
    """Plot sample interest rate paths.

    Creates a visualization of the simulated interest rate paths, showing how
    the short rate evolves over time for different scenarios.

    Args:
        paths: Array of shape (num_paths, num_steps) containing the simulated paths
        dt: Time step size used in the simulation
        n: Number of paths to display in the plot

    Returns:
        Figure: Matplotlib figure object containing the path visualization
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    maturities = np.arange(1, paths.shape[1] + 1) * dt
    for p in paths[:n]:
        ax.plot(maturities, p)

    ax.set_xlabel("Years", labelpad=10)
    ax.set_ylabel("Short Rate", labelpad=10)
    ax.xaxis.grid(alpha=0.7, linewidth=0.5)
    ax.yaxis.grid(alpha=0.7, linewidth=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{100 * x:.2f}%"))
    ax.set_title("Sample Paths", fontsize=20, pad=15)
    fig.tight_layout()
    return fig


def initialize_paths(
    maxT: float, dt: float, num_paths: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Initialize arrays for Monte Carlo path simulation.

    Creates the necessary arrays for storing random increments and their cumulative
    sums, along with the time steps for the simulation.

    Args:
        maxT: Maximum time horizon for the simulation
        dt: Time step size
        num_paths: Number of paths to simulate

    Returns:
        Tuple containing:
        - dW: Array of random increments (num_paths × num_steps)
        - cumsum_dW: Cumulative sum of random increments
        - time_steps: Array of time points for the simulation
    """
    num_steps = int(maxT / dt)
    dW = np.random.randn(num_paths, num_steps)
    cumsum_dW = np.cumsum(dW, axis=1)
    time_steps = np.arange(1, num_steps + 1) * dt
    return dW, cumsum_dW, time_steps


def price_monte_carlo_merton(
    model: Merton, T: float, maxT: float, dt: float, num_paths: int = 10_000
) -> float:
    """Calculate zero-coupon bond price using Monte Carlo simulation for the Merton model.

    The Merton model assumes the short rate follows a Brownian motion with drift:
    dr(t) = μdt + σdW(t)

    The simulation uses the exact solution for the Merton model:
    r(t) = r₀ + μt + σW(t)

    Args:
        model: A Merton model instance containing the model parameters
        T: Time to maturity in years
        maxT: Maximum time horizon for the simulation
        dt: Time step size
        num_paths: Number of paths to simulate (default: 10,000)

    Returns:
        float: The zero-coupon bond price

    Note:
        For very short maturities (T < dt), returns the exact solution exp(-r₀T)
    """
    r0, mu, sigma = model.r0, model.mu, model.sigma
    steps = int(T / dt)
    if steps == 0:
        return math.exp(-r0 * T)

    _, cumsum_dW, time_steps = initialize_paths(maxT, dt, num_paths)
    paths = r0 + mu * time_steps[:steps] + sigma * np.sqrt(dt) * cumsum_dW[:, :steps]
    discount_factors = np.exp(-np.sum(dt * paths, axis=1))
    return float(np.mean(discount_factors))


def plot_paths_merton(model: Merton, T: float, maxT: float, dt: float, n: int = 10) -> Figure:
    """Plot sample paths for the Merton model.

    Args:
        model: A Merton model instance
        T: Time to maturity
        maxT: Maximum time horizon
        dt: Time step size
        n: Number of paths to display (default: 10)

    Returns:
        Figure: Matplotlib figure showing the simulated paths
    """
    r0, mu, sigma = model.r0, model.mu, model.sigma
    steps = int(maxT / dt)
    _, cumsum_dW, time_steps = initialize_paths(maxT, dt, n)
    paths = r0 + mu * time_steps[:steps] + sigma * np.sqrt(dt) * cumsum_dW[:, :steps]
    return plot_paths(paths, dt, n)


def price_monte_carlo_vasicek(
    model: Vasicek, T: float, maxT: float, dt: float, num_paths: int = 10_000
) -> float:
    """Calculate zero-coupon bond price using Monte Carlo simulation for the Vasicek model.

    The Vasicek model assumes the short rate follows an Ornstein-Uhlenbeck process:
    dr(t) = κ(θ - r(t))dt + σdW(t)

    The simulation uses Euler-Maruyama discretization:
    r(t+dt) = r(t) + κ(θ - r(t))dt + σ√dt * dW(t)

    Args:
        model: A Vasicek model instance containing the model parameters
        T: Time to maturity in years
        maxT: Maximum time horizon for the simulation
        dt: Time step size
        num_paths: Number of paths to simulate (default: 10,000)

    Returns:
        float: The zero-coupon bond price

    Note:
        For very short maturities (T < dt), returns the exact solution exp(-r₀T)
    """
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    steps = int(T / dt)
    if steps == 0:
        return math.exp(-r0 * T)

    dW, _, _ = initialize_paths(maxT, dt, num_paths)
    paths = np.empty((num_paths, steps + 1))
    paths[:, 0] = r0
    for t in range(steps):
        paths[:, t + 1] = (
            paths[:, t] + kappa * (theta - paths[:, t]) * dt + np.sqrt(dt) * sigma * dW[:, t]
        )

    discount_factors = np.exp(-np.sum(dt * paths[:, :steps], axis=1))
    return float(np.mean(discount_factors))


def plot_paths_vasicek(model: Vasicek, T: float, maxT: float, dt: float, n: int = 10) -> Figure:
    """Plot sample paths for the Vasicek model.

    Args:
        model: A Vasicek model instance
        T: Time to maturity
        maxT: Maximum time horizon
        dt: Time step size
        n: Number of paths to display (default: 10)

    Returns:
        Figure: Matplotlib figure showing the simulated paths
    """
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    steps = int(maxT / dt)
    dW, _, _ = initialize_paths(maxT, dt, n)
    paths = np.empty((n, steps + 1))
    paths[:, 0] = r0
    for t in range(steps):
        paths[:, t + 1] = (
            paths[:, t] + kappa * (theta - paths[:, t]) * dt + np.sqrt(dt) * sigma * dW[:, t]
        )
    return plot_paths(paths, dt, n)


def price_monte_carlo_g2(
    model: G2, T: float, maxT: float, dt: float, num_paths: int = 1_000
) -> float:
    """Calculate zero-coupon bond price using Monte Carlo simulation for the G2 model.

    The G2 model is a two-factor Gaussian model where the short rate is:
    r(t) = x(t) + y(t) + φ

    where x(t) and y(t) follow correlated Ornstein-Uhlenbeck processes:
    dx(t) = -ax(t)dt + σₓdW₁(t)
    dy(t) = -by(t)dt + σᵧdW₂(t)
    dW₁(t)dW₂(t) = ρdt

    The simulation uses Euler-Maruyama discretization for both factors.

    Args:
        model: A G2 model instance containing the model parameters
        T: Time to maturity in years
        maxT: Maximum time horizon for the simulation
        dt: Time step size
        num_paths: Number of paths to simulate (default: 1,000)

    Returns:
        float: The zero-coupon bond price

    Note:
        For very short maturities (T < dt), returns the exact solution exp(-(x₀ + y₀ + φ)T)
        The integral of the short rate is clipped to [-100, 100] to prevent numerical overflow
    """
    x0, y0, a, b, rho, phi, sigma_x, sigma_y = (
        model.x0,
        model.y0,
        model.a,
        model.b,
        model.rho,
        model.phi,
        model.sigma_x,
        model.sigma_y,
    )
    steps = int(T / dt)
    if steps == 0:
        return math.exp(-(x0 + y0 + phi) * T)

    dWx, _, _ = initialize_paths(maxT, dt, num_paths)
    dWy, _, _ = initialize_paths(maxT, dt, num_paths)

    x_paths = np.empty((num_paths, steps + 1))
    y_paths = np.empty((num_paths, steps + 1))
    x_paths[:, 0] = x0
    y_paths[:, 0] = y0
    for t in range(steps):
        dx = -a * x_paths[:, t] * dt + sigma_x * np.sqrt(dt) * dWx[:, t]
        dy = (
            -b * y_paths[:, t] * dt
            + sigma_y * rho * np.sqrt(dt) * dWx[:, t]
            + sigma_y * np.sqrt(1 - rho**2) * np.sqrt(dt) * dWy[:, t]
        )
        x_paths[:, t + 1] = x_paths[:, t] + dx
        y_paths[:, t + 1] = y_paths[:, t] + dy

    paths = x_paths + y_paths + phi
    integral = np.sum(dt * paths[:, :], axis=1)
    integral = np.clip(integral, -100, 100)
    discount_factors = np.exp(-integral)
    return float(np.mean(discount_factors))


def plot_paths_g2(model: G2, T: float, maxT: float, dt: float, n: int = 10) -> Figure:
    """Plot sample paths for the G2 model.

    Args:
        model: A G2 model instance
        T: Time to maturity
        maxT: Maximum time horizon
        dt: Time step size
        n: Number of paths to display (default: 10)

    Returns:
        Figure: Matplotlib figure showing the simulated paths
    """
    x0, y0, a, b, rho, phi, sigma_x, sigma_y = (
        model.x0,
        model.y0,
        model.a,
        model.b,
        model.rho,
        model.phi,
        model.sigma_x,
        model.sigma_y,
    )
    steps = int(maxT / dt)
    dWx, _, _ = initialize_paths(maxT, dt, n)
    dWy, _, _ = initialize_paths(maxT, dt, n)

    x_paths = np.empty((n, steps + 1))
    y_paths = np.empty((n, steps + 1))
    x_paths[:, 0] = x0
    y_paths[:, 0] = y0
    for t in range(steps):
        dx = -a * x_paths[:, t] * dt + sigma_x * np.sqrt(dt) * dWx[:, t]
        dy = (
            -b * y_paths[:, t] * dt
            + sigma_y * rho * np.sqrt(dt) * dWx[:, t]
            + sigma_y * np.sqrt(1 - rho**2) * np.sqrt(dt) * dWy[:, t]
        )
        x_paths[:, t + 1] = x_paths[:, t] + dx
        y_paths[:, t + 1] = y_paths[:, t] + dy

    paths = x_paths + y_paths + phi
    return plot_paths(paths, dt, n)
