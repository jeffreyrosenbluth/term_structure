"""Gaussian Volatility Two-Factor Plus (GV2P) model for interest rates.

This module implements the Gaussian Volatility Two-Factor Plus (GV2P) model, which
is an extension of the two-factor Gaussian model that includes a third factor to
better capture the term structure of interest rates. The model combines two
correlated Gaussian factors with a deterministic component.

The model is characterized by the following stochastic differential equations:
dx(t) = -λ*x(t)dt + σₓdW₁(t)
dy(t) = -γ*y(t)dt + σᵧdW₂(t)
dz(t) = -k*z(t)dt

where:
- x(t), y(t) are the stochastic factors
- z(t) is the deterministic factor
- λ, γ are the mean reversion speeds
- k is the decay rate for the deterministic factor
- σₓ, σᵧ are the volatility parameters
- W₁(t), W₂(t) are correlated Brownian motions
- φ is a scaling parameter

The short rate is given by r(t) = x(t) + y(t) + φ*z(t).
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class GV2P(Model):
    """Gaussian Volatility Two-Factor Plus (GV2P) model for interest rates.

    The GV2P model extends the two-factor Gaussian model by adding a deterministic
    component that helps capture the term structure more accurately. The model
    combines two correlated Gaussian factors with a deterministic decay component.

    Attributes:
        x0: Initial value of the first stochastic factor
        y0: Initial value of the second stochastic factor
        z0: Initial value of the deterministic factor
        lambda_: Mean reversion speed of the first factor (λ > 0)
        gamma: Mean reversion speed of the second factor (γ > 0)
        k: Decay rate for the deterministic factor (k > 0)
        phi: Scaling parameter for the deterministic component
        sigma_x: Volatility parameter for the first factor
        sigma_y: Volatility parameter for the second factor
        _sigma_x_bounds: Optional bounds for sigma_x when sigma_x_center is provided
        _sigma_y_bounds: Optional bounds for sigma_y when sigma_y_center is provided
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        z0: float,
        lambda_: float,
        gamma: float,
        k: float,
        phi: float,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        """Initialize the GV2P model.

        Args:
            x0: Initial value of the first stochastic factor
            y0: Initial value of the second stochastic factor
            z0: Initial value of the deterministic factor
            lambda_: Mean reversion speed of the first factor (λ > 0)
            gamma: Mean reversion speed of the second factor (γ > 0)
            k: Decay rate for the deterministic factor (k > 0)
            phi: Scaling parameter for the deterministic component
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
        self.z0 = z0
        self.lambda_ = lambda_
        self.gamma = gamma
        self.sigma_x: float = cast(float, sigma_x_center if sigma_x is None else sigma_x)
        self.sigma_y: float = cast(float, sigma_y_center if sigma_y is None else sigma_y)
        self.k = k
        self.phi = phi

        # Calculate bounds if center values are provided
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
            f"--- GV2P ---\n"
            f"x0={self.x0}\n"
            f"y0={self.y0}\n"
            f"z0={self.z0}\n"
            f"lambda_={self.lambda_}\n"
            f"gamma={self.gamma}\n"
            f"sigma_x={self.sigma_x}\n"
            f"sigma_y={self.sigma_y}\n"
            f"k={self.k}\n"
            f"phi={self.phi}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [x0, y0, z0, lambda_, gamma, sigma_x, sigma_y, k, phi]
        """
        return np.array(
            [
                self.x0,
                self.y0,
                self.z0,
                self.lambda_,
                self.gamma,
                self.sigma_x,
                self.sigma_y,
                self.k,
                self.phi,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, a: NDArray[np.float64]) -> "GV2P":
        """Create a new model instance from a numpy array of parameters.

        Args:
            a: Array containing [x0, y0, z0, lambda_, gamma, sigma_x, sigma_y, k, phi]

        Returns:
            GV2P: A new instance of the model with parameters from the array
        """
        return cls(*a.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - x0, y0, z0: (-inf, inf)
        - lambda_, gamma: [0, inf)
        - sigma_x, sigma_y: [0.0001, inf)
        - k: [0.0001, inf)
        - phi: (-inf, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                0.0,
                0.0,
                0.0001,
                0.0001,
                0.0001,
                -np.inf,
            ]
        )
        upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
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
            lower[5] = self._sigma_x_bounds[0]
            upper[5] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[6] = self._sigma_y_bounds[0]
            upper[6] = self._sigma_y_bounds[1]

        return lower, upper

    def params(self) -> "GV2P":
        """Get the current model parameters.

        Returns:
            GV2P: The current model instance
        """
        return self

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another GV2P model instance to copy parameters from

        Raises:
            AssertionError: If p is not a GV2P model instance
        """
        assert isinstance(p, GV2P)
        self.x0 = p.x0
        self.y0 = p.y0
        self.z0 = p.z0
        self.lambda_ = p.lambda_
        self.gamma = p.gamma
        self.sigma_x = p.sigma_x
        self.sigma_y = p.sigma_y
        self.k = p.k
        self.phi = p.phi
