from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray


def spot_rate(
    price: Union[float, Sequence[float], NDArray[np.float64]],
    T: Union[float, Sequence[float], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Calculate spot rates from bond prices and maturities.

    Args:
        price: Bond price(s). Can be a single float or array of prices.
        T: Time to maturity in years. Can be a single float or array of maturities.

    Returns:
        NDArray[np.float64]: Spot rate(s) calculated as -ln(price)/T

    Raises:
        ValueError: If T contains zero values or if price contains negative values
    """
    price_arr = np.asarray(price, dtype=float)
    T_arr = np.asarray(T, dtype=float)

    if np.any(T_arr == 0.0):
        raise ValueError("T cannot be (or contain) zero")
    if np.any(price_arr <= 0.0):
        raise ValueError("Price cannot be (or contain) zero or negative values")

    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log(price_arr) / T_arr
    return np.where(np.isfinite(y), y, np.inf)
