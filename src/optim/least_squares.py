"""Least squares optimization module.

This module provides a wrapper around SciPy's least squares optimization functionality
for fitting interest rate models to market data. The module implements a class that
handles the optimization process using SciPy's `least_squares` function, which is
particularly well-suited for fitting term structure models where the objective is
to minimize the sum of squared residuals between model and market prices/yields.

The optimization process involves:
1. Defining a residual function that computes the difference between model and market values
2. Setting up parameter bounds to ensure physically meaningful results
3. Running the optimization to find the best-fit parameters
4. Handling the optimization results and potential convergence issues
"""

from typing import Any, Callable, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

ArrayF = NDArray[np.float64]


class SciPyLeastSquares:
    """Wrapper for SciPy's least squares optimization.

    This class provides a convenient interface to SciPy's `least_squares` function,
    which implements the Levenberg-Marquardt algorithm for solving nonlinear least
    squares problems. It's particularly useful for fitting interest rate models
    where the objective is to minimize the sum of squared residuals between model
    and market prices/yields.

    The class allows for customization of the optimization process through keyword
    arguments passed to the underlying SciPy function, such as:
    - `ftol`: Function tolerance for convergence
    - `xtol`: Parameter tolerance for convergence
    - `max_nfev`: Maximum number of function evaluations
    - `method`: Optimization method ('trf', 'lm', or 'dogbox')

    Attributes:
        lsq_kwargs: Dictionary of keyword arguments passed to SciPy's least_squares
    """

    def __init__(self, **lsq_kwargs: Any) -> None:
        """Initialize the least squares optimizer.

        Args:
            **lsq_kwargs: Keyword arguments to be passed to SciPy's least_squares function.
                         Common options include:
                         - ftol: Function tolerance (default: 1e-8)
                         - xtol: Parameter tolerance (default: 1e-8)
                         - max_nfev: Maximum function evaluations (default: None)
                         - method: Optimization method (default: 'trf')
        """
        self.lsq_kwargs = lsq_kwargs

    def minimize(
        self,
        x0: Union[Sequence[float], NDArray[np.float64]],
        residuals: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        bounds: Union[Tuple[NDArray[np.float64], NDArray[np.float64]], None] = None,
    ) -> NDArray[np.float64]:
        """Minimize the sum of squared residuals.

        This method runs the optimization process to find the parameters that minimize
        the sum of squared residuals between model and market values. It uses SciPy's
        `least_squares` function under the hood, which implements the Levenberg-Marquardt
        algorithm.

        Args:
            x0: Initial guess for the parameters to be optimized. Can be a sequence
                of floats or a numpy array.
            residuals: Function that computes the residuals (differences between model
                      and market values) for a given set of parameters. Should return
                      a numpy array of residuals.
            bounds: Optional tuple of (lower_bounds, upper_bounds) for the parameters.
                   If None, no bounds are applied.

        Returns:
            NDArray[np.float64]: The optimized parameters that minimize the sum of
                                squared residuals.

        Note:
            The residuals function should be designed to return an array of differences
            between model and market values, which will be squared and summed by the
            optimization algorithm.
        """
        res = least_squares(
            residuals,
            x0,
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            **self.lsq_kwargs,
        )
        # mypy still thinks res.x is Any → cast it
        return cast(NDArray[np.float64], res.x)
