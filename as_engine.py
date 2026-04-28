"""
Avellaneda–Stoikov (AS) market making primitives.

This module implements the core closed-form quotes for the classic AS model under:
- Mid-price dynamics: dS_t = σ dW_t (zero drift Brownian motion)
- Exponential liquidity / fill intensity: λ(δ) = A * exp(-κ δ)
- CARA utility with risk aversion γ, finite horizon τ (time remaining)

References (common form):
- Reservation price: r = m - q * γ * σ^2 * τ
- Optimal spread:     s* = γ * σ^2 * τ + (2/γ) * ln(1 + γ/κ)
"""

from __future__ import annotations

import math


def calculate_reservation_price(
    mid_price: float,
    inventory: float,
    volatility: float,
    gamma: float,
    time_horizon: float,
) -> float:
    """
    Compute the AS reservation price.

    Math:
    - The dealer is indifferent (in expected utility terms) to buying or selling at the
      reservation price r. With CARA utility and Brownian mid-price risk, inventory q
      creates a risk penalty proportional to variance over the remaining horizon τ:

        Var[S_{t+τ} - S_t] = σ^2 τ

      This shifts the "fair" price away from mid m by:

        r = m - q * γ * σ^2 * τ

    Intuition:
    - If q > 0 (long), you want to sell, so r is BELOW mid (more aggressive ask, less aggressive bid).
    - If q < 0 (short), you want to buy, so r is ABOVE mid.
    """
    if volatility < 0:
        raise ValueError("volatility must be >= 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if time_horizon < 0:
        raise ValueError("time_horizon must be >= 0")

    sigma2 = float(volatility) ** 2
    tau = float(time_horizon)
    q = float(inventory)
    m = float(mid_price)

    return m - q * gamma * sigma2 * tau


def calculate_spread(
    volatility: float,
    gamma: float,
    time_horizon: float,
    kappa: float,
) -> float:
    """
    Compute the AS optimal (total) spread s*.

    Math:
    - In the AS model with exponential intensity λ(δ) = A e^{-κ δ}, the optimal quotes
      can be written around the reservation price r with symmetric half-spread δ*:

        bid = r - s*/2
        ask = r + s*/2

      The closed-form optimal TOTAL spread is:

        s* = γ σ^2 τ + (2/γ) * ln(1 + γ/κ)

      where:
      - γ σ^2 τ is the *inventory risk* term (wider spread when risk aversion, volatility, or horizon increase)
      - (2/γ) ln(1 + γ/κ) is the *market liquidity* term (tighter spreads when κ is large = fills drop quickly with distance)
    """
    if volatility < 0:
        raise ValueError("volatility must be >= 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if time_horizon < 0:
        raise ValueError("time_horizon must be >= 0")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")

    sigma2 = float(volatility) ** 2
    tau = float(time_horizon)

    risk_term = gamma * sigma2 * tau
    liquidity_term = (2.0 / gamma) * math.log(1.0 + (gamma / float(kappa)))

    return risk_term + liquidity_term


def get_quotes(
    mid_price: float,
    inventory: float,
    volatility: float,
    gamma: float,
    time_horizon: float,
    kappa: float,
) -> tuple[float, float]:
    """
    Compute (bid, ask) quotes using AS reservation price + optimal spread.

    Steps:
    - Compute reservation price r = m - q γ σ^2 τ
    - Compute optimal spread s* = γ σ^2 τ + (2/γ) ln(1 + γ/κ)
    - Place symmetric quotes around r:
        bid = r - s*/2
        ask = r + s*/2
    """
    r = calculate_reservation_price(
        mid_price=mid_price,
        inventory=inventory,
        volatility=volatility,
        gamma=gamma,
        time_horizon=time_horizon,
    )
    s = calculate_spread(
        volatility=volatility,
        gamma=gamma,
        time_horizon=time_horizon,
        kappa=kappa,
    )

    half = 0.5 * s
    bid = r - half
    ask = r + half
    return bid, ask


if __name__ == "__main__":
    # Simple self-test with example parameters.
    mid_price = 0.095
    inventory = 0.0
    volatility = 0.002
    gamma = 0.1
    time_horizon = 1.0
    kappa = 1.5

    r = calculate_reservation_price(
        mid_price=mid_price,
        inventory=inventory,
        volatility=volatility,
        gamma=gamma,
        time_horizon=time_horizon,
    )
    s = calculate_spread(
        volatility=volatility,
        gamma=gamma,
        time_horizon=time_horizon,
        kappa=kappa,
    )
    bid, ask = get_quotes(
        mid_price=mid_price,
        inventory=inventory,
        volatility=volatility,
        gamma=gamma,
        time_horizon=time_horizon,
        kappa=kappa,
    )

    print(f"Reservation price: {r:.8f}")
    print(f"Optimal spread:    {s:.8f}")
    print(f"Bid quote:         {bid:.8f}")
    print(f"Ask quote:         {ask:.8f}")

