"""Two-factor Gaussian (G2) model for interest rates.

This module implements the two-factor Gaussian (G2) model, which is a popular model
for interest rates that uses two correlated Gaussian factors to model the short rate.
The model is particularly useful for capturing the term structure of interest rates
and is analytically tractable.

The model is characterized by the following stochastic differential equations:
dx(t) = -a*x(t)dt + σₓdW₁(t)
dy(t) = -b*y(t)dt + σᵧdW₂(t)

where:
- x(t), y(t) are the two factors at time t
- a, b are the mean reversion speeds
- σₓ, σᵧ are the volatility parameters
- W₁(t), W₂(t) are correlated Brownian motions with correlation ρ
- φ is a scaling parameter

The short rate is given by r(t) = x(t) + φ*y(t).
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class G2(Model):
    """Two-factor Gaussian (G2) model for interest rates.

    The G2 model uses two correlated Gaussian factors to model the short rate,
    providing a flexible framework for capturing the term structure. The model
    is analytically tractable and allows for closed-form solutions for many
    interest rate derivatives.

    Attributes:
        x0: Initial value of the first factor
        y0: Initial value of the second factor
        a: Mean reversion speed of the first factor
        b: Mean reversion speed of the second factor
        rho: Correlation between the two Brownian motions (-1 ≤ ρ ≤ 1)
        phi: Scaling parameter for the second factor
        sigma_x: Volatility parameter for the first factor
        sigma_y: Volatility parameter for the second factor
        _sigma_x_bounds: Optional bounds for sigma_x when sigma_x_center is provided
        _sigma_y_bounds: Optional bounds for sigma_y when sigma_y_center is provided
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        a: float,
        b: float,
        rho: float,
        phi: float,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        """Initialize the G2 model.

        Args:
            x0: Initial value of the first factor
            y0: Initial value of the second factor
            a: Mean reversion speed of the first factor
            b: Mean reversion speed of the second factor
            rho: Correlation between the two Brownian motions (-1 ≤ ρ ≤ 1)
            phi: Scaling parameter for the second factor
            sigma_x: Volatility parameter for the first factor
            sigma_y: Volatility parameter for the second factor
            sigma_x_center: Optional center value for sigma_x, used to set bounds
            sigma_y_center: Optional center value for sigma_y, used to set bounds

        Raises:
            ValueError: If neither sigma_x nor sigma_x_center is provided, or
                      if neither sigma_y nor sigma_y_center is provided
        """
        if sigma_x is None and sigma_x_center is None:
            raise ValueError("Either sigma_x or sigma_x_center must be provided")
        if sigma_y is None and sigma_y_center is None:
            raise ValueError("Either sigma_y or sigma_y_center must be provided")

        self.x0 = x0
        self.y0 = y0
        self.a = a
        self.b = b
        self.rho = rho
        self.phi = phi
        self.sigma_x: float = cast(float, sigma_x_center if sigma_x is None else sigma_x)
        self.sigma_y: float = cast(float, sigma_y_center if sigma_y is None else sigma_y)
        self._sigma_x_bounds: Optional[Tuple[float, float]] = None
        self._sigma_y_bounds: Optional[Tuple[float, float]] = None

        if sigma_x_center is not None:
            self._sigma_x_bounds = (sigma_x_center * 0.95, sigma_x_center * 1.05)

        if sigma_y_center is not None:
            self._sigma_y_bounds = (sigma_y_center * 0.95, sigma_y_center * 1.05)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: Formatted string showing all model parameters
        """
        return (
            f"--- G2 ---\n"
            f"x0={self.x0}\n"
            f"y0={self.y0}\n"
            f"a={self.a}\n"
            f"b={self.b}\n"
            f"rho={self.rho}\n"
            f"phi={self.phi}\n"
            f"sigma_x={self.sigma_x}\n"
            f"sigma_y={self.sigma_y}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [x0, y0, a, b, rho, phi, sigma_x, sigma_y]
        """
        return np.array(
            [self.x0, self.y0, self.a, self.b, self.rho, self.phi, self.sigma_x, self.sigma_y],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "G2":  # type: ignore
        """Create a new model instance from a numpy array of parameters.

        Args:
            a: Array containing [x0, y0, a, b, rho, phi, sigma_x, sigma_y]

        Returns:
            G2: A new instance of the model with parameters from the array
        """
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - x0, y0: (-inf, inf)
        - a, b: (-inf, inf)
        - rho: [-1, 1]
        - phi: (-inf, inf)
        - sigma_x, sigma_y: [0.0001, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -1.0, -np.inf, 0.0001, 0.0001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf, 1.0, np.inf, np.inf, np.inf])
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the current parameter bounds for this model instance.

        If sigma_x_center or sigma_y_center was provided during initialization,
        their respective bounds will be set to (0.95 * center, 1.05 * center).
        Otherwise, returns the default bounds.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower, upper = self.bounds()

        if self._sigma_x_bounds is not None:
            lower[6] = self._sigma_x_bounds[0]
            upper[6] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[7] = self._sigma_y_bounds[0]
            upper[7] = self._sigma_y_bounds[1]

        return lower, upper

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another G2 model instance to copy parameters from

        Raises:
            AssertionError: If p is not a G2 model instance
        """
        assert isinstance(p, G2)
        self.x0 = p.x0
        self.y0 = p.y0
        self.a = p.a
        self.b = p.b
        self.rho = p.rho
        self.phi = p.phi
        self.sigma_x = p.sigma_x
        self.sigma_y = p.sigma_y
