import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.core.engine import PricingEngine
from src.core.model import ShortRateModel
from src.models.cir import CIRParams
from src.models.cir2 import CIR2Params
from src.models.g2 import G2Params
from src.models.gv2p import GV2PParams
from src.models.merton import MertonParams
from src.models.vasicek import VasicekParams


class ClosedFormMerton(PricingEngine[MertonParams]):
    def P(self, model: ShortRateModel[MertonParams], T: float) -> float:
        p = model.params()
        r0, mu, sigma = p.r0, p.mu, p.sigma
        return float(np.exp(-r0 * T - 0.5 * mu * T**2 + sigma**2 * T**3 / 6.0))


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


class ClosedFormG2(PricingEngine[G2Params]):
    def P(self, model: ShortRateModel[G2Params], T: float) -> float:
        p = model.params()
        x0, y0, a, b, sigma_x, sigma_y, rho, phi = (
            p.x0,
            p.y0,
            p.a,
            p.b,
            p.sigma_x,
            p.sigma_y,
            p.rho,
            p.phi,
        )
        term1 = (sigma_x**2 / a**2) * (
            T + (2 / a) * np.exp(-a * T) - (1 / (2 * a)) * np.exp(-2 * a * T) - (3 / (2 * a))
        )
        term2 = (sigma_y**2 / b**2) * (
            T + (2 / b) * np.exp(-b * T) - (1 / (2 * b)) * np.exp(-2 * b * T) - (3 / (2 * b))
        )
        term3 = (2 * rho * sigma_x * sigma_y / (a * b)) * (
            T
            + (np.exp(-a * T) - 1) / a
            + (np.exp(-b * T) - 1) / b
            - (np.exp(-(a + b) * T) - 1) / (a + b)
        )
        V = term1 + term2 + term3
        A = -phi * T - x0 * (1 - np.exp(-a * T)) / a - y0 * (1 - np.exp(-b * T)) / b + 0.5 * V

        # Add numerical safeguards to prevent overflow
        if A > 700:  # np.exp(700) is approximately 1e304, close to float64 max
            A = 700
        elif A < -700:  # np.exp(-700) is approximately 1e-304, close to float64 min
            A = -700

        return float(np.exp(A))


class ClosedFormCIR2(PricingEngine[CIR2Params]):
    def P(self, model: ShortRateModel[CIR2Params], T: float) -> float:
        def _cir_AB(
            kappa: float, theta: float, sigma: float, t: NDArray[np.float64]
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
            gamma = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
            exp_gt = np.exp(gamma * t)
            denom = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma
            B = 2.0 * (exp_gt - 1.0) / denom
            A = (2.0 * gamma * np.exp((gamma + kappa) * t / 2.0) / denom) ** (
                2.0 * kappa * theta / (sigma * sigma)
            )
            return A, B

        p: CIR2Params = model.params()
        A1, B1 = _cir_AB(p.kappa1, p.theta1, p.sigma_x, T)
        A2, B2 = _cir_AB(p.kappa2, p.theta2, p.sigma_y, T)

        return float(A1 * A2 * np.exp(-B1 * p.r0_1 - B2 * p.r0_2))


class ClosedFormGV2P(PricingEngine[GV2PParams]):
    def P(self, model: ShortRateModel[GV2PParams], T: float) -> float:
        p = model.params()
        x0, y0, z0, lambda_, gamma, sigma_x, sigma_y, k, phi = (
            p.x0,
            p.y0,
            p.z0,
            p.lambda_,
            p.gamma,
            p.sigma_x,
            p.sigma_y,
            p.k,
            p.phi,
        )
        B = (1 - np.exp(-k * T)) / k
        C = T - (1 - np.exp(-k * T)) / k
        if abs(k - gamma) < 1e-10:  # Handle case where k is very close to gamma
            D = (1 - np.exp(-gamma * T)) / gamma - T * np.exp(-gamma * T)
        else:
            D = (1 - np.exp(-gamma * T)) / gamma - (
                k * (1 - np.exp(-gamma * T)) - gamma * (1 - np.exp(-k * T))
            ) / (k * gamma * (k - gamma))
        A1 = -phi * T
        A2 = -lambda_ * (T**2 / 2 - T / k + (1 - np.exp(-k * T)) / (k**2))
        A3 = -(sigma_x**2) * (T**3 / 6)
        A4 = -(sigma_y**2) * (T / (2 * gamma**2) - (1 - np.exp(-gamma * T)) / (gamma**3))
        A = A1 + A2 + A3 + A4
        return float(np.exp(A - B * z0 - C * x0 - D * y0))
