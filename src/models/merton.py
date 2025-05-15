"""Merton model for interest rates.

This module implements the Merton model, which is a simple model for interest rates
where the short rate follows a Brownian motion with drift. The model is characterized
by three parameters: initial rate (r0), drift (mu), and volatility (sigma).
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class Merton(Model):
    """Merton model for interest rates.

    The Merton model assumes that the short rate follows a Brownian motion with drift:
    dr(t) = mu*dt + sigma*dW(t)

    where:
    - r(t) is the short rate at time t
    - mu is the drift parameter
    - sigma is the volatility parameter
    - W(t) is a standard Brownian motion

    Attributes:
        r0: Initial short rate
        mu: Drift parameter
        sigma: Volatility parameter
        _sigma_bounds: Optional bounds for sigma parameter when sigma_center is provided
    """

    def __init__(
        self,
        r0: float,
        mu: float,
        sigma: Optional[float] = None,
        sigma_center: Optional[float] = None,
    ) -> None:
        """Initialize the Merton model.

        Args:
            r0: Initial short rate
            mu: Drift parameter
            sigma: Volatility parameter
            sigma_center: Optional center value for sigma, used to set bounds

        Raises:
            ValueError: If neither sigma nor sigma_center is provided
        """
        if sigma is None and sigma_center is None:
            raise ValueError("Either sigma or sigma_center must be provided")

        self.r0 = r0
        self.mu = mu
        self.sigma: float = cast(float, sigma_center if sigma is None else sigma)
        self._sigma_bounds: Optional[Tuple[float, float]] = None

        if sigma_center is not None:
            self._sigma_bounds = (sigma_center * 0.95, sigma_center * 1.05)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: Formatted string showing all model parameters
        """
        return f"--- Merton ---\nr0={self.r0}\nmu={self.mu}\nsigma={self.sigma}\n"

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [r0, mu, sigma]
        """
        return np.array([self.r0, self.mu, self.sigma], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Merton":  # type: ignore
        """Create a new model instance from a numpy array of parameters.

        Args:
            arr: Array containing [r0, mu, sigma]

        Returns:
            Merton: A new instance of the model with parameters from the array
        """
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - r0: (-inf, inf)
        - mu: (-inf, inf)
        - sigma: (0.001, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array([-np.inf, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf])
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
            lower[2] = self._sigma_bounds[0]
            upper[2] = self._sigma_bounds[1]

        return lower, upper

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another Merton model instance to copy parameters from

        Raises:
            AssertionError: If p is not a Merton model instance
        """
        assert isinstance(p, Merton)
        self.r0 = p.r0
        self.mu = p.mu
        self.sigma = p.sigma
