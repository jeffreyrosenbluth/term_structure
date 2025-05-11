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


def _initialize_paths(
    maxT: float, dt: float, num_paths: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    num_steps = int(maxT / dt)
    dW = np.random.randn(num_paths, num_steps)
    cumsum_dW = np.cumsum(dW, axis=1)
    time_steps = np.arange(1, num_steps + 1) * dt
    return dW, cumsum_dW, time_steps


def price_monte_carlo_merton(
    model: Merton, T: float, maxT: float, dt: float, num_paths: int = 10_000
) -> float:
    r0, mu, sigma = model.r0, model.mu, model.sigma
    steps = int(T / dt)
    if steps == 0:
        return math.exp(-r0 * T)

    _, cumsum_dW, time_steps = _initialize_paths(maxT, dt, num_paths)
    paths = r0 + mu * time_steps[:steps] + sigma * np.sqrt(dt) * cumsum_dW[:, :steps]
    discount_factors = np.exp(-np.sum(dt * paths, axis=1))
    return float(np.mean(discount_factors))


def plot_monte_carlo_merton(model: Merton, T: float, maxT: float, dt: float, n: int = 10) -> Figure:
    r0, mu, sigma = model.r0, model.mu, model.sigma
    steps = int(T / dt)
    _, cumsum_dW, time_steps = _initialize_paths(maxT, dt, n)
    paths = r0 + mu * time_steps[:steps] + sigma * np.sqrt(dt) * cumsum_dW[:, :steps]
    return plot_paths(paths, dt, n)


def price_monte_carlo_vasicek(
    model: Vasicek, T: float, maxT: float, dt: float, num_paths: int = 10_000
) -> float:
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    steps = int(T / dt)
    if steps == 0:
        return math.exp(-r0 * T)

    dW, _, _ = _initialize_paths(maxT, dt, num_paths)
    paths = np.empty((num_paths, steps + 1))
    paths[:, 0] = r0
    for t in range(steps):
        paths[:, t + 1] = (
            paths[:, t] + kappa * (theta - paths[:, t]) * dt + np.sqrt(dt) * sigma * dW[:, t]
        )

    discount_factors = np.exp(-np.sum(dt * paths[:, :steps], axis=1))
    return float(np.mean(discount_factors))


def plot_monte_carlo_vasicek(
    model: Vasicek, T: float, maxT: float, dt: float, n: int = 10
) -> Figure:
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    steps = int(T / dt)
    dW, _, _ = _initialize_paths(maxT, dt, n)
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

    dWx, _, _ = _initialize_paths(maxT, dt, num_paths)
    dWy, _, _ = _initialize_paths(maxT, dt, num_paths)

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


def plot_monte_carlo_g2(model: G2, T: float, maxT: float, dt: float, n: int = 10) -> Figure:
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
    dWx, _, _ = _initialize_paths(maxT, dt, n)
    dWy, _, _ = _initialize_paths(maxT, dt, n)

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
