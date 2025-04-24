import math
from typing import TypeVar

import numpy as np

from src.core.engine import PricingEngine
from src.core.model import ShortRateModel
from src.core.parameter import Parameters
from src.models.cir import CIRParams
from src.models.g2 import G2Params
from src.models.vasicek import VasicekParams

P = TypeVar("P", bound=Parameters)


class ClosedFormVasicek(PricingEngine[VasicekParams]):
    def P(self, model: ShortRateModel[VasicekParams], T: float) -> float:
        p = model.params()
        r0, kappa, theta, sigma = p.r0, p.kappa, p.theta, p.sigma
        B = (1 - math.exp(-kappa * T)) / kappa
        A = math.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 * B**2 / (4 * kappa))
        return A * math.exp(-B * r0)


class ClosedFormCIR(PricingEngine[CIRParams]):
    def P(self, model: ShortRateModel[CIRParams], T: float) -> float:
        p = model.params()
        r0, kappa, theta, sigma = p.r0, p.kappa, p.theta, p.sigma

        h = np.sqrt(kappa**2 + 2 * sigma**2)
        exp_hT = np.exp((h * T))
        exp_term_A = np.exp((kappa + h) * T / 2)

        term_in_b_denom = (kappa + h) * (exp_hT - 1)
        b_denom = 2 * h + term_in_b_denom
        B_T = 2 * (exp_hT - 1) / b_denom

        a_base_num = 2 * h * exp_term_A
        a_base_den = b_denom
        a_base = a_base_num / a_base_den

        a_exponent = (2 * kappa * theta) / sigma**2
        A_T = np.power(a_base, a_exponent)

        exp_arg = -B_T * r0
        exp_term_P = np.exp(exp_arg)
        return float(A_T * exp_term_P)


# Engine ----------------------------------------------------------------
class ClosedFormG2(PricingEngine[G2Params]):
    def P(self, model: ShortRateModel[G2Params], T: float) -> float:
        def _B(h: float, t: float) -> float:
            return (1.0 - math.exp(-h * t)) / h if h > 1e-12 else t

        p = model.params()
        a, b, sx, sy, rho, phi = p.a, p.b, p.sigma_x, p.sigma_y, p.rho, p.phi

        B_a = _B(a, T)
        B_b = _B(b, T)

        # Prevent overflow in variance calculations
        var_x = sx**2 / (2 * a) * (1.0 - np.exp(-2 * a * T))
        var_y = sy**2 / (2 * b) * (1.0 - np.exp(-2 * b * T))
        cov_xy = rho * sx * sy / (a + b) * (1.0 - np.exp(-(a + b) * T))

        V = (B_a**2) * var_x + (B_b**2) * var_y + 2 * B_a * B_b * cov_xy

        # Clip the exponent to prevent overflow
        exponent = -phi * T + 0.5 * V
        max_exp = 700  # np.exp(700) is still finite
        clipped_exp = np.clip(exponent, -max_exp, max_exp)

        return float(np.exp(clipped_exp))
