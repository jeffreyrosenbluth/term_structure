from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore


class SciPyLeastSquares:
    """Adapter → our internal `LeastSquares` Protocol."""

    def __init__(
        self, bounds: Optional[Sequence[Tuple[float, float]]] = None, **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        if bounds is None:
            self.bounds = (-np.inf, np.inf)
        else:
            self.bounds = (
                np.array([b[0] for b in bounds], dtype=np.float64),
                np.array([b[1] for b in bounds], dtype=np.float64),
            )

    def minimize(
        self,
        x0: NDArray[np.float64],
        residuals: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        res = least_squares(residuals, x0, bounds=self.bounds, **self.kwargs)
        return res.x
