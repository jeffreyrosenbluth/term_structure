from typing import Any, Callable, Dict, Generic, Sequence, Tuple, Type, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from src.core.model import Model
from src.core.solver import SciPyLeastSquares
from src.engines.closed_form import (
    price_closed_form_cir,
    price_closed_form_cir2,
    price_closed_form_g2,
    price_closed_form_g2plus,
    price_closed_form_gv2p,
    price_closed_form_merton,
    price_closed_form_v2,
    price_closed_form_vasicek,
)
from src.engines.monte_carlo import (
    price_monte_carlo_g2,
    price_monte_carlo_merton,
    price_monte_carlo_vasicek,
)
from src.models.cir import CIR
from src.models.cir2 import CIR2
from src.models.g2 import G2
from src.models.g2_plus import G2plus
from src.models.gv2p import GV2P
from src.models.merton import Merton
from src.models.v2 import V2
from src.models.vasicek import Vasicek

P = TypeVar("P", bound=Model)

# Map model types to their corresponding closed-form pricing functions
CLOSED_FORM_ENGINES: Dict[Type[Model], Callable[[Model, float], float]] = {
    cast(Type[Model], Merton): cast(Callable[[Model, float], float], price_closed_form_merton),
    cast(Type[Model], Vasicek): cast(Callable[[Model, float], float], price_closed_form_vasicek),
    cast(Type[Model], CIR): cast(Callable[[Model, float], float], price_closed_form_cir),
    cast(Type[Model], G2): cast(Callable[[Model, float], float], price_closed_form_g2),
    cast(Type[Model], GV2P): cast(Callable[[Model, float], float], price_closed_form_gv2p),
    cast(Type[Model], V2): cast(Callable[[Model, float], float], price_closed_form_v2),
    cast(Type[Model], CIR2): cast(Callable[[Model, float], float], price_closed_form_cir2),
    cast(Type[Model], G2plus): cast(Callable[[Model, float], float], price_closed_form_g2plus),
}

# Map model types to their corresponding Monte Carlo pricing functions
MONTE_CARLO_ENGINES: Dict[Type[Model], Callable[[Model, float, float, float, int], float]] = {
    cast(Type[Model], Merton): cast(
        Callable[[Model, float, float, float, int], float], price_monte_carlo_merton
    ),
    cast(Type[Model], Vasicek): cast(
        Callable[[Model, float, float, float, int], float], price_monte_carlo_vasicek
    ),
    cast(Type[Model], G2): cast(
        Callable[[Model, float, float, float, int], float], price_monte_carlo_g2
    ),
}


class Calibrator(Generic[P]):
    def __init__(
        self,
        model: P,
        engine: Callable[[P, float], float],
        solver: Any,
    ) -> None:
        self.model = model
        self.engine = engine
        self.solver = solver

    def calibrate(self, market: Sequence[Tuple[float, float]]) -> None:
        t, y = map(np.asarray, zip(*market))

        def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            params_cls = type(self.model)
            self.model.update(params_cls.from_array(theta))
            prices = np.array([self.engine(self.model, ti) for ti in t])
            return -np.log(prices) / t - y  # Convert prices to yields

        lower, upper = self.model.get_bounds()

        theta0 = self.model.to_array()
        theta_star = self.solver.minimize(
            theta0,
            residuals,
            bounds=(lower, upper),
        )
        params_cls = type(self.model)
        self.model.update(params_cls.from_array(theta_star))

    @staticmethod
    def calibrate_model(
        model_cls: Type[P],
        engine: Callable[[P, float], float],
        solver: Any,
        market_data: Sequence[Tuple[float, float]],
        **model_kwargs: float,
    ) -> Tuple[P, Callable[[P, float], float]]:
        """Helper function to create, calibrate model and calculate prices/yields.

        Args:
            model_cls: Model class (e.g., Vasicek)
            engine: Pricing function that takes (model, T) and returns price
            solver: Optimization solver
            market_data: Sequence of (maturity, yield) pairs
            **model_kwargs: Initial parameters for the model

        Returns:
            Tuple of (calibrated model, engine)
        """
        model = model_cls(**model_kwargs)
        calibrator = Calibrator(model, engine, solver)
        calibrator.calibrate(market_data)

        return model, engine

    @staticmethod
    def calibrate_closed_form(
        model_cls: Type[P],
        market_data: Sequence[Tuple[float, float]],
        **model_kwargs: float,
    ) -> Tuple[P, Callable[[P, float], float]]:
        """Helper function to create, calibrate model and calculate prices/yields using closed-form engine.

        Args:
            model_cls: Model class (e.g., Vasicek)
            market_data: Sequence of (maturity, yield) pairs
            **model_kwargs: Initial parameters for the model

        Returns:
            Tuple of (calibrated model, engine)
        """
        if model_cls not in CLOSED_FORM_ENGINES:
            raise ValueError(f"No closed-form engine available for {model_cls.__name__}")

        return Calibrator.calibrate_model(
            model_cls,
            cast(Callable[[P, float], float], CLOSED_FORM_ENGINES[model_cls]),
            SciPyLeastSquares(),
            market_data,
            **model_kwargs,
        )

    @staticmethod
    def calibrate_monte_carlo(
        model_cls: Type[P],
        market_data: Sequence[Tuple[float, float]],
        maxT: float,
        dt: float,
        num_paths: int = 10_000,
        **model_kwargs: float,
    ) -> Tuple[P, Callable[[P, float], float]]:
        """Helper function to create, calibrate model and calculate prices/yields using Monte Carlo engine.

        Args:
            model_cls: Model class (e.g., Vasicek)
            market_data: Sequence of (maturity, yield) pairs
            maxT: Maximum time horizon for simulation
            dt: Time step size for simulation
            num_paths: Number of Monte Carlo paths
            **model_kwargs: Initial parameters for the model

        Returns:
            Tuple of (calibrated model, engine)
        """
        if model_cls not in MONTE_CARLO_ENGINES:
            raise ValueError(f"No Monte Carlo engine available for {model_cls.__name__}")

        mc_engine = cast(
            Callable[[P, float, float, float, int], float], MONTE_CARLO_ENGINES[model_cls]
        )

        def engine(model: P, T: float) -> float:
            return mc_engine(model, T, maxT, dt, num_paths)

        return Calibrator.calibrate_model(
            model_cls,
            engine,
            SciPyLeastSquares(),
            market_data,
            **model_kwargs,
        )
