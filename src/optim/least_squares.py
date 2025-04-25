from typing import Any, Callable, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # type: ignore[import-untyped]

ArrayF = NDArray[np.float64]


class SciPyLeastSquares:
    def __init__(self, **lsq_kwargs: Any) -> None:
        self.lsq_kwargs = lsq_kwargs

    def minimize(
        self,
        x0: Union[Sequence[float], NDArray[np.float64]],
        residuals: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        bounds: Union[Tuple[NDArray[np.float64], NDArray[np.float64]], None] = None,
    ) -> NDArray[np.float64]:  # precise return type
        res = least_squares(
            residuals,
            x0,
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            **self.lsq_kwargs,
        )
        # mypy still thinks res.x is Any → cast it
        return cast(NDArray[np.float64], res.x)
