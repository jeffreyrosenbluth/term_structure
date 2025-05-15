"""Vasicek model for interest rates.

This module implements the Vasicek model, which is one of the earliest and most
influential models for interest rates. It is a mean-reverting Gaussian model that
assumes the short rate follows an Ornstein-Uhlenbeck process.

The model is characterized by the following stochastic differential equation:
dr(t) = κ(θ - r(t))dt + σdW(t)

where:
- r(t) is the short rate at time t
- κ is the mean reversion speed (κ > 0)
- θ is the long-term mean level
- σ is the volatility parameter
- W(t) is a standard Brownian motion

The model allows for negative interest rates and has closed-form solutions for
many interest rate derivatives.
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class Vasicek(Model):
    """Vasicek model for interest rates.

    The Vasicek model is a mean-reverting Gaussian model that assumes the short
    rate follows an Ornstein-Uhlenbeck process. It is analytically tractable and
    provides a good balance between mathematical tractability and empirical fit.

    Attributes:
        r0: Initial short rate
        kappa: Mean reversion speed (κ > 0)
        theta: Long-term mean level
        sigma: Volatility parameter
        _sigma_bounds: Optional bounds for sigma when sigma_center is provided
    """

    def __init__(
        self,
        r0: float,
        kappa: float,
        theta: float,
        sigma: Optional[float] = None,
        sigma_center: Optional[float] = None,
    ) -> None:
        """Initialize the Vasicek model.

        Args:
            r0: Initial short rate
            kappa: Mean reversion speed (κ > 0)
            theta: Long-term mean level
            sigma: Volatility parameter
            sigma_center: Optional center value for sigma, used to set bounds

        Raises:
            ValueError: If neither sigma nor sigma_center is provided
        """
        if sigma is None and sigma_center is None:
            raise ValueError("Either sigma or sigma_center must be provided")

        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma: float = cast(float, sigma_center if sigma is None else sigma)
        self._sigma_bounds: Optional[Tuple[float, float]] = None

        if sigma_center is not None:
            self._sigma_bounds = (sigma_center * 0.95, sigma_center * 1.05)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: Formatted string showing all model parameters
        """
        return (
            f"--- Vasicek ---\n"
            f"r0={self.r0}\n"
            f"kappa={self.kappa}\n"
            f"theta={self.theta}\n"
            f"sigma={self.sigma}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [r0, kappa, theta, sigma]
        """
        return np.array([self.r0, self.kappa, self.theta, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Vasicek":  # type: ignore
        """Create a new model instance from a numpy array of parameters.

        Args:
            arr: Array containing [r0, kappa, theta, sigma]

        Returns:
            Vasicek: A new instance of the model with parameters from the array
        """
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - r0: (-inf, inf)
        - kappa: [0, inf)
        - theta: (-inf, inf)
        - sigma: [0.001, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array([-np.inf, 0.000, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the current parameter bounds for this model instance.

        If sigma_center was provided during initialization, the sigma bounds will be
        set to (0.95 * sigma_center, 1.05 * sigma_center). Otherwise, returns the
        default bounds.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower, upper = self.bounds()

        if self._sigma_bounds is not None:
            lower[3] = self._sigma_bounds[0]
            upper[3] = self._sigma_bounds[1]

        return lower, upper

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another Vasicek model instance to copy parameters from

        Raises:
            AssertionError: If p is not a Vasicek model instance
        """
        assert isinstance(p, Vasicek)
        self.r0 = p.r0
        self.kappa = p.kappa
        self.theta = p.theta
        self.sigma = p.sigma
        self._sigma_bounds = p._sigma_bounds
