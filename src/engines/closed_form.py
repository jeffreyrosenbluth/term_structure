import math
from typing import Tuple

import numpy as np

from src.models.cir import CIR
from src.models.cir2 import CIR2
from src.models.g2 import G2
from src.models.g2_plus import G2plus
from src.models.gv2p import GV2P
from src.models.merton import Merton
from src.models.v2 import V2
from src.models.vasicek import Vasicek


def price_closed_form_merton(model: Merton, T: float) -> float:
    r0, mu, sigma = model.r0, model.mu, model.sigma
    return float(np.exp(-r0 * T - 0.5 * mu * T**2 + sigma**2 * T**3 / 6.0))


def price_closed_form_vasicek(model: Vasicek, T: float) -> float:
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    B = (1 - math.exp(-kappa * T)) / kappa
    A = math.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 * B**2 / (4 * kappa))
    return A * math.exp(-B * r0)


def price_closed_form_cir(model: CIR, T: float) -> float:
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma

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


def price_closed_form_g2(model: G2, T: float) -> float:
    x0, y0, a, b, rho, phi, sigma_x, sigma_y = (
        model.x0,
        model.y0,
        model.a,
        model.b,
        model.rho,
        model.phi,
        model.sigma_x,
        model.sigma_y,
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

    # Numerical safeguards to prevent overflow
    if A > 700:  # np.exp(700) is approximately 1e304, close to float64 max
        A = 700
    elif A < -700:  # np.exp(-700) is approximately 1e-304, close to float64 min
        A = -700

    return float(np.exp(A))


def price_closed_form_cir2(model: CIR2, T: float) -> float:
    def _cir_AB(kappa: float, theta: float, sigma: float, t: float) -> Tuple[float, float]:
        gamma = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
        exp_gt = math.exp(gamma * t)
        denom = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma
        B = 2.0 * (exp_gt - 1.0) / denom
        A = (2.0 * gamma * math.exp((gamma + kappa) * t / 2.0) / denom) ** (
            2.0 * kappa * theta / (sigma * sigma)
        )
        return A, B

    A1, B1 = _cir_AB(model.kappa1, model.theta1, model.sigma_x, T)
    A2, B2 = _cir_AB(model.kappa2, model.theta2, model.sigma_y, T)

    # Add numerical safeguards to prevent overflow
    exp_term = -B1 * model.r0_1 - B2 * model.r0_2
    if exp_term > 700:
        exp_term = 700
    elif exp_term < -700:
        exp_term = -700

    return float(A1 * A2 * math.exp(exp_term))


def price_closed_form_gv2p(model: GV2P, T: float) -> float:
    x0, y0, z0, lambda_, gamma, sigma_x, sigma_y, k, phi = (
        model.x0,
        model.y0,
        model.z0,
        model.lambda_,
        model.gamma,
        model.sigma_x,
        model.sigma_y,
        model.k,
        model.phi,
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

    # Add numerical safeguards to prevent overflow
    exp_arg = A - B * z0 - C * x0 - D * y0
    if exp_arg > 700:  # np.exp(700) is approximately 1e304, close to float64 max
        exp_arg = 700
    elif exp_arg < -700:  # np.exp(-700) is approximately 1e-304, close to float64 min
        exp_arg = -700

    return float(np.exp(exp_arg))


def price_closed_form_v2(model: V2, T: float) -> float:
    y1_0, y2_0, k11, k21, k22, delta0, delta1, delta2, sigma1, sigma2 = (
        model.y1_0,
        model.y2_0,
        model.k11,
        model.k21,
        model.k22,
        model.delta0,
        model.delta1,
        model.delta2,
        model.sigma1,
        model.sigma2,
    )
    B2 = (delta2 / k22) * (1 - np.exp(-k22 * T))
    B1 = (delta1 / k11) * (1 - np.exp(-k11 * T))
    if abs(k11 - k22) > 1e-8:
        B1 += (k21 * delta2) / (k22 * (k11 - k22)) * (np.exp(-k22 * T) - np.exp(-k11 * T))
    else:  # special case k11 == k22
        B1 += (k21 * delta2) / (k22**2) * T * np.exp(-k22 * T)

    I2 = (delta2 / k22) ** 2 * (
        T - 2 / k22 * (1 - np.exp(-k22 * T)) + 1 / (2 * k22) * (1 - np.exp(-2 * k22 * T))
    )
    # Approximate I1
    I1 = (delta1 / k11) ** 2 * (
        T - 2 / k11 * (1 - np.exp(-k11 * T)) + 1 / (2 * k11) * (1 - np.exp(-2 * k11 * T))
    )
    A = -delta0 * T + 0.5 * (sigma1**2 * I1 + sigma2**2 * I2)
    logP = A + B1 * y1_0 + B2 * y2_0

    return float(np.exp(logP))


def price_closed_form_g2plus(model: G2plus, T: float) -> float:
    x0, y0, z0, a, b, rho, phi, k, sigma_x, sigma_y = (
        model.x0,
        model.y0,
        model.z0,
        model.a,
        model.b,
        model.rho,
        model.phi,
        model.k,
        model.sigma_x,
        model.sigma_y,
    )
    Bz = (1 - np.exp(-k * T)) / T
    Bx = (1 - np.exp(-a * T)) / a - (np.exp(-k * T) - np.exp(-a * T)) / (a - k)
    By = (1 - np.exp(-b * T)) / b - (np.exp(-k * T) - np.exp(-b * T)) / (b - k)

    def I(a: float, b: float) -> float:
        t1 = (1 - np.exp(-(a + b) * T)) / ((a + b) * a * b)
        t2 = (1 - np.exp(-(a + k) * T)) / ((a + k) * a * (a - k))
        t3 = (1 - np.exp(-(b + k) * T)) / ((b + k) * b * (b - k))
        t4 = (1 - np.exp(-2 * k * T)) / (2 * k * (a - k) * (b - k))
        return float(t1 + t2 + t3 + t4)

    V = sigma_x**2 * I(a, a) + sigma_y**2 * I(b, b) + 2 * rho * sigma_x * sigma_y * I(a, b)
    A = -phi * T + 0.5 * V
    exp_arg = A - Bx * T * x0 - By * T + y0 - Bz * T * z0

    if exp_arg > 700:  # np.exp(700) is approximately 1e304, close to float64 max
        exp_arg = 700
    elif exp_arg < -700:  # np.exp(-700) is approximately 1e-304, close to float64 min
        exp_arg = -700

    return float(np.exp(exp_arg))
