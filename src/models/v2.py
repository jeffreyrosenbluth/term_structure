"""Two-factor Vasicek (V2) model for interest rates.

This module implements the two-factor Vasicek (V2) model, which is an extension
of the single-factor Vasicek model that uses two correlated factors to model the
short rate. The model allows for a more flexible term structure while maintaining
analytical tractability.

The model is characterized by the following stochastic differential equations:
dy₁(t) = -k₁₁y₁(t)dt + σ₁dW₁(t)
dy₂(t) = -(k₂₁y₁(t) + k₂₂y₂(t))dt + σ₂dW₂(t)

where:
- y₁(t), y₂(t) are the two factors at time t
- k₁₁, k₂₁, k₂₂ are the mean reversion parameters
- σ₁, σ₂ are the volatility parameters
- W₁(t), W₂(t) are correlated Brownian motions
- δ₀, δ₁, δ₂ are the constant and factor loadings

The short rate is given by r(t) = δ₀ + δ₁y₁(t) + δ₂y₂(t).
"""

from typing import Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model


class V2(Model):
    """Two-factor Vasicek (V2) model for interest rates.

    The V2 model extends the single-factor Vasicek model by adding a second factor
    that can be correlated with the first factor. This allows for a more flexible
    term structure while maintaining analytical tractability.

    Attributes:
        y1_0: Initial value of the first factor
        y2_0: Initial value of the second factor
        k11: Mean reversion speed of the first factor (k₁₁ > 0)
        k21: Cross-factor mean reversion parameter
        k22: Mean reversion speed of the second factor (k₂₂ > 0)
        delta0: Constant term in the short rate equation
        delta1: Loading of the first factor
        delta2: Loading of the second factor
        sigma1: Volatility parameter for the first factor
        sigma2: Volatility parameter for the second factor
        _sigma1_bounds: Optional bounds for sigma1 when sigma1_center is provided
        _sigma2_bounds: Optional bounds for sigma2 when sigma2_center is provided
    """

    def __init__(
        self,
        y1_0: float,
        y2_0: float,
        k11: float,
        k21: float,
        k22: float,
        delta0: float,
        delta1: float,
        delta2: float,
        sigma1: Optional[float] = None,
        sigma2: Optional[float] = None,
        sigma1_center: Optional[float] = None,
        sigma2_center: Optional[float] = None,
    ) -> None:
        """Initialize the V2 model.

        Args:
            y1_0: Initial value of the first factor
            y2_0: Initial value of the second factor
            k11: Mean reversion speed of the first factor (k₁₁ > 0)
            k21: Cross-factor mean reversion parameter
            k22: Mean reversion speed of the second factor (k₂₂ > 0)
            delta0: Constant term in the short rate equation
            delta1: Loading of the first factor
            delta2: Loading of the second factor
            sigma1: Volatility parameter for the first factor
            sigma2: Volatility parameter for the second factor
            sigma1_center: Optional center value for sigma1, used to set bounds
            sigma2_center: Optional center value for sigma2, used to set bounds

        Raises:
            ValueError: If neither sigma1 nor sigma1_center is provided, or
                      if neither sigma2 nor sigma2_center is provided
        """
        if sigma1 is None and sigma1_center is None:
            raise ValueError("Either sigma1 or sigma1_center must be provided")
        if sigma2 is None and sigma2_center is None:
            raise ValueError("Either sigma2 or sigma2_center must be provided")

        self.y1_0 = y1_0
        self.y2_0 = y2_0
        self.k11 = k11
        self.k21 = k21
        self.k22 = k22
        self.delta0 = delta0
        self.delta1 = delta1
        self.delta2 = delta2
        self.sigma1: float = cast(float, sigma1_center if sigma1 is None else sigma1)
        self.sigma2: float = cast(float, sigma2_center if sigma2 is None else sigma2)
        self._sigma1_bounds: Optional[Tuple[float, float]] = None
        self._sigma2_bounds: Optional[Tuple[float, float]] = None

        if sigma1_center is not None:
            self._sigma1_bounds = (sigma1_center * 0.95, sigma1_center * 1.05)

        if sigma2_center is not None:
            self._sigma2_bounds = (sigma2_center * 0.95, sigma2_center * 1.05)

    def __str__(self) -> str:
        """Return a string representation of the model parameters.

        Returns:
            str: Formatted string showing all model parameters
        """
        return (
            f"--- V2 ---\n"
            f"y1_0={self.y1_0}\n"
            f"y2_0={self.y2_0}\n"
            f"k11={self.k11}\n"
            f"k21={self.k21}\n"
            f"k22={self.k22}\n"
            f"delta0={self.delta0}\n"
            f"delta1={self.delta1}\n"
            f"delta2={self.delta2}\n"
            f"sigma1={self.sigma1}\n"
            f"sigma2={self.sigma2}\n"
        )

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing [y1_0, y2_0, k11, k21, k22, delta0, delta1, delta2, sigma1, sigma2]
        """
        return np.array(
            [
                self.y1_0,
                self.y2_0,
                self.k11,
                self.k21,
                self.k22,
                self.delta0,
                self.delta1,
                self.delta2,
                self.sigma1,
                self.sigma2,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "V2":
        """Create a new model instance from a numpy array of parameters.

        Args:
            arr: Array containing [y1_0, y2_0, k11, k21, k22, delta0, delta1, delta2, sigma1, sigma2]

        Returns:
            V2: A new instance of the model with parameters from the array
        """
        return cls(*arr.tolist())

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        The bounds are:
        - y1_0, y2_0: (-inf, inf)
        - k11, k22: [0, inf)
        - k21: (-inf, inf)
        - delta0, delta1, delta2: (-inf, inf)
        - sigma1, sigma2: [0.001, inf)

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower = np.array(
            [-np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf, -np.inf, -np.inf, 0.001, 0.001]
        )
        upper = np.array(
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        )
        return lower, upper

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the current parameter bounds for this model instance.

        If sigma1_center or sigma2_center was provided during initialization,
        their respective bounds will be set to (0.95 * center, 1.05 * center).
        Otherwise, returns the default bounds.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
        """
        lower, upper = self.bounds()

        if self._sigma1_bounds is not None:
            lower[8] = self._sigma1_bounds[0]
            upper[8] = self._sigma1_bounds[1]

        if self._sigma2_bounds is not None:
            lower[9] = self._sigma2_bounds[0]
            upper[9] = self._sigma2_bounds[1]

        return lower, upper

    def params(self) -> "V2":
        """Get the current model parameters.

        Returns:
            V2: The current model instance
        """
        return self

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another V2 model instance to copy parameters from

        Raises:
            AssertionError: If p is not a V2 model instance
        """
        assert isinstance(p, V2)
        self.y1_0 = p.y1_0
        self.y2_0 = p.y2_0
        self.k11 = p.k11
        self.k21 = p.k21
        self.k22 = p.k22
        self.delta0 = p.delta0
        self.delta1 = p.delta1
        self.delta2 = p.delta2
        self.sigma1 = p.sigma1
        self.sigma2 = p.sigma2
