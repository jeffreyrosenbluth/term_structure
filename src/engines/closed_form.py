"""Closed-form solutions for interest rate models.

This module provides closed-form solutions for zero-coupon bond prices under various
interest rate models. Each function implements the analytical solution for a specific
model, allowing for fast and accurate price calculations without numerical methods.

The module includes solutions for:
- Merton model (Brownian motion with drift)
- Vasicek model (Ornstein-Uhlenbeck process)
- CIR model (Cox-Ingersoll-Ross)
- G2 model (Two-factor Gaussian)
- CIR2 model (Two-factor CIR)
- GV2P model (Gaussian-Vasicek-2-Plus)
- V2 model (Two-factor Vasicek)
- G2plus model (Two-factor Gaussian Plus)
"""

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
    """Calculate the zero-coupon bond price using the Merton model's closed-form solution.

    The Merton model assumes the short rate follows a Brownian motion with drift:
    dr(t) = μdt + σdW(t)

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = exp(-r₀T - 0.5μT² + σ²T³/6)

    where:
    - r₀ is the initial short rate
    - μ is the drift parameter
    - σ is the volatility parameter
    - T is the time to maturity

    Args:
        model: A Merton model instance containing the model parameters (r0, mu, sigma)
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
    r0, mu, sigma = model.r0, model.mu, model.sigma
    return float(np.exp(-r0 * T - 0.5 * mu * T**2 + sigma**2 * T**3 / 6.0))


def price_closed_form_vasicek(model: Vasicek, T: float) -> float:
    """Calculate the zero-coupon bond price using the Vasicek model's closed-form solution.

    The Vasicek model assumes the short rate follows an Ornstein-Uhlenbeck process:
    dr(t) = κ(θ - r(t))dt + σdW(t)

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = A(T)exp(-B(T)r₀)

    where:
    - r₀ is the initial short rate
    - κ is the mean reversion speed
    - θ is the long-term mean level
    - σ is the volatility parameter
    - T is the time to maturity
    - A(T) and B(T) are functions of the model parameters and T

    Args:
        model: A Vasicek model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
    r0, kappa, theta, sigma = model.r0, model.kappa, model.theta, model.sigma
    B = (1 - math.exp(-kappa * T)) / kappa
    A = math.exp((theta - sigma**2 / (2 * kappa**2)) * (B - T) - sigma**2 * B**2 / (4 * kappa))
    return A * math.exp(-B * r0)


def price_closed_form_cir(model: CIR, T: float) -> float:
    """Calculate the zero-coupon bond price using the CIR model's closed-form solution.

    The CIR model assumes the short rate follows:
    dr(t) = κ(θ - r(t))dt + σ√r(t)dW(t)

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = A(T)exp(-B(T)r₀)

    where:
    - r₀ is the initial short rate
    - κ is the mean reversion speed
    - θ is the long-term mean level
    - σ is the volatility parameter
    - T is the time to maturity
    - A(T) and B(T) are functions of the model parameters and T

    Args:
        model: A CIR model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
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
    """Calculate the zero-coupon bond price using the G2 model's closed-form solution.

    The G2 model is a two-factor Gaussian model where the short rate is:
    r(t) = x(t) + y(t) + φ

    where x(t) and y(t) follow correlated Ornstein-Uhlenbeck processes:
    dx(t) = -ax(t)dt + σₓdW₁(t)
    dy(t) = -by(t)dt + σᵧdW₂(t)
    dW₁(t)dW₂(t) = ρdt

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = exp(A(T) - Bₓ(T)x₀ - Bᵧ(T)y₀)

    where:
    - x₀, y₀ are the initial values of the factors
    - a, b are the mean reversion speeds
    - σₓ, σᵧ are the volatility parameters
    - ρ is the correlation between the factors
    - φ is the constant term
    - T is the time to maturity
    - A(T), Bₓ(T), Bᵧ(T) are functions of the model parameters and T

    Args:
        model: A G2 model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
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
    """Calculate the zero-coupon bond price using the CIR2 model's closed-form solution.

    The CIR2 model is a two-factor CIR model where the short rate is:
    r(t) = r₁(t) + r₂(t)

    where r₁(t) and r₂(t) follow independent CIR processes:
    dr₁(t) = κ₁(θ₁ - r₁(t))dt + σₓ√r₁(t)dW₁(t)
    dr₂(t) = κ₂(θ₂ - r₂(t))dt + σᵧ√r₂(t)dW₂(t)

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = A₁(T)A₂(T)exp(-B₁(T)r₁₀ - B₂(T)r₂₀)

    where:
    - r₁₀, r₂₀ are the initial values of the factors
    - κ₁, κ₂ are the mean reversion speeds
    - θ₁, θ₂ are the long-term mean levels
    - σₓ, σᵧ are the volatility parameters
    - T is the time to maturity
    - A₁(T), A₂(T), B₁(T), B₂(T) are functions of the model parameters and T

    Args:
        model: A CIR2 model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """

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
    """Calculate the zero-coupon bond price using the GV2P model's closed-form solution.

    The GV2P model is a three-factor model where the short rate is:
    r(t) = x(t) + y(t) + z(t) + φ

    where:
    - x(t) follows a Vasicek process
    - y(t) follows a CIR process
    - z(t) follows a deterministic process

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = exp(A(T) - B(T)z₀ - C(T)x₀ - D(T)y₀)

    where:
    - x₀, y₀, z₀ are the initial values of the factors
    - λ is the drift parameter for z(t)
    - γ is the mean reversion speed for y(t)
    - σₓ, σᵧ are the volatility parameters
    - k is the mean reversion speed for x(t)
    - φ is the constant term
    - T is the time to maturity
    - A(T), B(T), C(T), D(T) are functions of the model parameters and T

    Args:
        model: A GV2P model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
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
    """Calculate the zero-coupon bond price using the V2 model's closed-form solution.

    The V2 model is a two-factor Vasicek model where the short rate is:
    r(t) = y₁(t) + y₂(t) + δ₀

    where y₁(t) and y₂(t) follow correlated Vasicek processes:
    dy₁(t) = -k₁₁y₁(t)dt + σ₁dW₁(t)
    dy₂(t) = (-k₂₁y₁(t) - k₂₂y₂(t))dt + σ₂dW₂(t)

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = exp(A(T) + B₁(T)y₁₀ + B₂(T)y₂₀)

    where:
    - y₁₀, y₂₀ are the initial values of the factors
    - k₁₁, k₂₁, k₂₂ are the mean reversion parameters
    - δ₀, δ₁, δ₂ are the constant terms
    - σ₁, σ₂ are the volatility parameters
    - T is the time to maturity
    - A(T), B₁(T), B₂(T) are functions of the model parameters and T

    Args:
        model: A V2 model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
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
    """Calculate the zero-coupon bond price using the G2plus model's closed-form solution.

    The G2plus model is a three-factor model where the short rate is:
    r(t) = x(t) + y(t) + z(t) + φ

    where:
    - x(t) and y(t) follow correlated Ornstein-Uhlenbeck processes
    - z(t) follows a deterministic process

    The closed-form solution for the zero-coupon bond price P(t,T) is:
    P(t,T) = exp(A(T) - Bₓ(T)x₀ - Bᵧ(T)y₀ - Bz(T)z₀)

    where:
    - x₀, y₀, z₀ are the initial values of the factors
    - a, b are the mean reversion speeds for x(t) and y(t)
    - ρ is the correlation between x(t) and y(t)
    - φ is the constant term
    - k is the mean reversion speed for z(t)
    - σₓ, σᵧ are the volatility parameters
    - T is the time to maturity
    - A(T), Bₓ(T), Bᵧ(T), Bz(T) are functions of the model parameters and T

    Args:
        model: A G2plus model instance containing the model parameters
        T: Time to maturity in years

    Returns:
        float: The zero-coupon bond price
    """
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

    def integral_term(a: float, b: float) -> float:
        t1 = (1 - np.exp(-(a + b) * T)) / ((a + b) * a * b)
        t2 = (1 - np.exp(-(a + k) * T)) / ((a + k) * a * (a - k))
        t3 = (1 - np.exp(-(b + k) * T)) / ((b + k) * b * (b - k))
        t4 = (1 - np.exp(-2 * k * T)) / (2 * k * (a - k) * (b - k))
        return float(t1 + t2 + t3 + t4)

    V = (
        sigma_x**2 * integral_term(a, a)
        + sigma_y**2 * integral_term(b, b)
        + 2 * rho * sigma_x * sigma_y * integral_term(a, b)
    )
    A = -phi * T + 0.5 * V
    exp_arg = A - Bx * T * x0 - By * T + y0 - Bz * T * z0

    if exp_arg > 700:  # np.exp(700) is approximately 1e304, close to float64 max
        exp_arg = 700
    elif exp_arg < -700:  # np.exp(-700) is approximately 1e-304, close to float64 min
        exp_arg = -700

    return float(np.exp(exp_arg))
