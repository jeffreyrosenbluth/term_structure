from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore


class SciPyLeastSquares:
    def minimize(
        self,
        x0: NDArray[np.float64],
        fun: Any,
        bounds: Tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Minimize a function using scipy's least_squares.

        Args:
            x0: Initial guess
            fun: Function to minimize
            bounds: Tuple of (lower, upper) bounds

        Returns:
            Optimized parameters
        """
        result = least_squares(
            fun,
            x0,
            bounds=bounds,
            method="trf",
            max_nfev=1000,
        )
        return result.x
