"""Two-factor Cox-Ingersoll-Ross (CIR2) model for interest rates.

This module implements the two-factor Cox-Ingersoll-Ross (CIR2) model, which is an
extension of the single-factor CIR model that uses two independent CIR processes
to model the short rate. This allows for more flexibility in capturing the term
structure of interest rates.

The model is characterized by the following stochastic differential equations:
dr₁(t) = κ₁(θ₁ - r₁(t))dt + σₓ√r₁(t)dW₁(t)
dr₂(t) = κ₂(θ₂ - r₂(t))dt + σᵧ√r₂(t)dW₂(t)

where:
- r₁(t), r₂(t) are the two factors at time t
- κ₁, κ₂ are the mean reversion speeds
- θ₁, θ₂ are the long-term mean levels
- σₓ, σᵧ are the volatility parameters
- W₁(t), W₂(t) are independent standard Brownian motions

The short rate is given by r(t) = r₁(t) + r₂(t).
Each factor ensures positivity when 2κᵢθᵢ ≥ σᵢ² (Feller condition).
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class CIR2(Model):
    """Two-factor Cox-Ingersoll-Ross (CIR2) model for interest rates.

    The CIR2 model combines two independent CIR processes to create a more flexible
    term structure model. Each factor follows its own mean-reverting square-root
    process, allowing for better fitting of market data.

    Attributes:
        r0_1: Initial value of the first factor
        r0_2: Initial value of the second factor
        kappa1: Mean reversion speed of the first factor
        theta1: Long-term mean level of the first factor
        kappa2: Mean reversion speed of the second factor
        theta2: Long-term mean level of the second factor
        sigma_x: Volatility parameter for the first factor
        sigma_y: Volatility parameter for the second factor
        _sigma_x_bounds: Optional bounds for sigma_x when sigma_x_center is provided
        _sigma_y_bounds: Optional bounds for sigma_y when sigma_y_center is provided
    """

    def __init__(
        self,
        r0_1: float,
        r0_2: float,
        kappa1: float,
        theta1: float,
        kappa2: float,
        theta2: float,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_x_center: Optional[float] = None,
        sigma_y_center: Optional[float] = None,
    ) -> None:
        """Initialize the CIR2 model.

        Args:
            r0_1: Initial value of the first factor
            r0_2: Initial value of the second factor
            kappa1: Mean reversion speed of the first factor
            theta1: Long-term mean level of the first factor
            kappa2: Mean reversion speed of the second factor
            theta2: Long-term mean level of the second factor
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

        self.r0_1 = r0_1
        self.r0_2 = r0_2
        self.kappa1 = kappa1
        self.theta1 = theta1
        self.kappa2 = kappa2
        self.theta2 = theta2
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
            f"--- CIR2 ---\n"
            f"r0_1={self.r0_1}\n"
            f"r0_2={self.r0_2}\n"
            f"kappa1={self.kappa1}\n"
            f"theta1={self.theta1}\n"
            f"sigma_x={self.sigma_x}\n"
            f"kappa2={self.kappa2}\n"
            f"theta2={self.theta2}\n"
            f"sigma_y={self.sigma_y}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [r0_1, r0_2, kappa1, theta1, kappa2, theta2, sigma_x, sigma_y]
        """
        return np.array(
            [
                self.r0_1,
                self.r0_2,
                self.kappa1,
                self.theta1,
                self.kappa2,
                self.theta2,
                self.sigma_x,
                self.sigma_y,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "CIR2":  # type: ignore
        """Create a new model instance from a numpy array of parameters.

        Args:
            arr: Array containing [r0_1, r0_2, kappa1, theta1, kappa2, theta2, sigma_x, sigma_y]

        Returns:
            CIR2: A new instance of the model with parameters from the array
        """
        return cls(
            r0_1=float(arr[0]),
            r0_2=float(arr[1]),
            kappa1=float(arr[2]),
            theta1=float(arr[3]),
            kappa2=float(arr[4]),
            theta2=float(arr[5]),
            sigma_x=float(arr[6]),
            sigma_y=float(arr[7]),
        )

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - r0_1, r0_2: (-inf, inf)
        - kappa1, kappa2: (-inf, inf)
        - theta1, theta2: (-inf, inf)
        - sigma_x, sigma_y: [0.001, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.001, -np.inf, -np.inf, 0.001])
        upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
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
            lower[4] = self._sigma_x_bounds[0]
            upper[4] = self._sigma_x_bounds[1]

        if self._sigma_y_bounds is not None:
            lower[7] = self._sigma_y_bounds[0]
            upper[7] = self._sigma_y_bounds[1]

        return lower, upper

    def params(self) -> "CIR2":
        """Get the current model parameters.

        Returns:
            CIR2: The current model instance
        """
        return self

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another CIR2 model instance to copy parameters from

        Raises:
            AssertionError: If p is not a CIR2 model instance
        """
        assert isinstance(p, CIR2)
        self.r0_1 = p.r0_1
        self.r0_2 = p.r0_2
        self.kappa1 = p.kappa1
        self.theta1 = p.theta1
        self.sigma_x = p.sigma_x
        self.kappa2 = p.kappa2
        self.theta2 = p.theta2
        self.sigma_y = p.sigma_y
