"""Monte Carlo simulation for interest rate models.

This module provides functionality for Monte Carlo simulation of interest rate paths
and calculating bond prices using these simulated paths. The simulation uses a simple
Euler-Maruyama discretization scheme.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MonteCarlo:
    """A class for Monte Carlo simulation of interest rate paths.

    This class stores simulated interest rate paths and provides methods to calculate
    bond prices using these paths. The paths are stored in a matrix where each row
    represents a different simulation path and each column represents a time step.

    Attributes:
        dt: Time step size used in the simulation
        paths: Matrix of simulated interest rate paths, shape (num_paths, num_steps)
    """

    dt: float
    paths: NDArray[np.float64]

    def price(self, T: float) -> float:
        """Calculate the price of a zero-coupon bond with maturity T.

        The price is calculated as the average of exp(-integral(r(t)dt)) across all
        simulated paths, where r(t) is the interest rate at time t.

        Args:
            T: Time to maturity in years

        Returns:
            float: The price of the zero-coupon bond

        Note:
            The price calculation assumes that the paths matrix contains enough time
            steps to cover the maturity T. If T/dt is greater than the number of
            columns in paths, the calculation will use all available time steps.
        """
        return float(np.mean(np.exp(-np.sum(self.dt * self.paths[:, : int(T / self.dt)], axis=1))))
