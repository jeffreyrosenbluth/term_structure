"""Base protocol for interest rate models.

This module defines the Model protocol that all interest rate models must implement.
The protocol ensures that models can be converted to/from arrays for optimization,
have parameter bounds defined, and can be updated with new parameters.
"""

from __future__ import annotations  # enables typing.Self on 3.11+

from typing import Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class Model(Protocol):
    """Protocol defining the interface that all interest rate models must implement.

    This protocol ensures that models can be:
    1. Converted to/from arrays for optimization purposes, i.e. handle models with different number
        of parameters
    2. Have parameter bounds defined for calibration
    3. Be updated with new parameters

    All interest rate models in the library must implement this protocol to ensure
    compatibility with the calibration and pricing infrastructure.
    """

    def to_array(self) -> NDArray[np.float64]:
        """Convert model parameters to a numpy array.

        Returns:
            NDArray[np.float64]: Array containing all model parameters in a fixed order
        """
        ...

    @classmethod
    def from_array(cls: type["Model"], a: NDArray[np.float64]) -> "Model":
        """Create a new model instance from a numpy array of parameters.

        Args:
            a: Array containing model parameters in the same order as to_array()

        Returns:
            Model: A new instance of the model with parameters from the array
        """
        ...

    @classmethod
    def bounds(cls) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the default parameter bounds for the model.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
            arrays defining the valid range for each parameter
        """
        ...

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get the current parameter bounds for this model instance.

        This method may return different bounds than the class method if the model
        has been configured with specific constraints.

        Returns:
            Tuple[NDArray[np.float64], NDArray[np.float64]]: Tuple of (lower_bounds, upper_bounds)
            arrays defining the valid range for each parameter
        """
        ...

    def update(self, p: "Model") -> None:
        """Update this model's parameters with those from another model.

        Args:
            p: Another model instance of the same type to copy parameters from
        """
        ...
