"""Causal long/cash BRK.B strategy candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

STRATEGY_NAMES = (
    "defensive_probability_overlay",
    "trend_filter",
    "trend_probability_hybrid",
    "support_pullback",
)


def defensive_probability_overlay(
    probabilities: pd.DataFrame,
    *,
    partial_spread: float = 0.08,
    cash_spread: float = 0.18,
) -> pd.Series:
    """Stay long by default and de-risk only on strong downside probability."""
    spread = probabilities["lower_first"] - probabilities["upper_first"]
    exposure = np.select(
        [spread >= cash_spread, spread >= partial_spread],
        [0.0, 0.5],
        default=1.0,
    )
    return pd.Series(exposure, index=probabilities.index, name="target_exposure")


def trend_filter(
    bars: pd.DataFrame,
    dates: pd.Index,
    *,
    band: float = 0.01,
) -> pd.Series:
    """Use a causal 200-day trend with a hysteresis band to reduce whipsaw."""
    trend = bars["Close"] / bars["Close"].rolling(200, min_periods=100).mean() - 1.0
    aligned = trend.reindex(dates)
    state = 1.0
    output: list[float] = []
    for value in aligned:
        if np.isfinite(value):
            if value > band:
                state = 1.0
            elif value < -band:
                state = 0.0
        output.append(state)
    return pd.Series(output, index=dates, name="target_exposure")


def trend_probability_hybrid(
    bars: pd.DataFrame,
    probabilities: pd.DataFrame,
    minimum_risk: pd.Series,
    *,
    probability_spread: float = 0.10,
    severe_minimum: float = -0.08,
) -> pd.Series:
    """Blend trend, first-touch probabilities, and empirical path-low risk."""
    dates = probabilities.index
    trend = bars["Close"] >= bars["Close"].rolling(200, min_periods=100).mean()
    trend = trend.reindex(dates).fillna(True)
    downside_lead = probabilities["lower_first"] - probabilities["upper_first"]
    risk = minimum_risk.reindex(dates).fillna(-0.05)
    exposure = pd.Series(np.where(trend, 1.0, 0.5), index=dates, dtype=float)
    moderate = (downside_lead >= probability_spread) | (risk <= severe_minimum * 0.75)
    severe = (downside_lead >= probability_spread * 1.75) | (risk <= severe_minimum)
    exposure.loc[moderate] = 0.5
    exposure.loc[severe] = 0.0
    return exposure.rename("target_exposure")


def support_pullback(
    bars: pd.DataFrame,
    probabilities: pd.DataFrame,
    levels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    support_distance: float = 0.035,
    resistance_distance: float = 0.025,
) -> pd.Series:
    """Retain uptrend pullbacks and reduce only on confirmed resistance risk."""
    dates = probabilities.index
    close = bars["Close"].reindex(dates)
    trend = (
        bars["Close"] >= bars["Close"].rolling(200, min_periods=100).mean()
    ).reindex(dates).fillna(True)
    support_gap = (close - levels["support"].reindex(dates)) / close
    resistance_gap = (levels["resistance"].reindex(dates) - close) / close
    rsi = features["rsi_14"].reindex(dates)
    momentum = features["momentum_20"].reindex(dates)
    downside_lead = probabilities["lower_first"] > probabilities["upper_first"]

    exposure = pd.Series(np.where(trend, 1.0, 0.25), index=dates, dtype=float)
    constructive_pullback = (
        trend
        & (support_gap <= support_distance)
        & (rsi >= rsi.shift(1))
        & (momentum > -0.08)
    )
    risky_resistance = (
        (resistance_gap <= resistance_distance)
        & downside_lead
        & (rsi < rsi.shift(1))
    )
    exposure.loc[constructive_pullback] = 1.0
    exposure.loc[risky_resistance] = 0.25
    return exposure.rename("target_exposure")


def strategy_parameter_grid() -> dict[str, list[dict[str, float]]]:
    """Small prespecified grids tuned only on the development period."""
    return {
        "defensive_probability_overlay": [
            {"partial_spread": 0.05, "cash_spread": 0.14},
            {"partial_spread": 0.08, "cash_spread": 0.18},
            {"partial_spread": 0.12, "cash_spread": 0.22},
        ],
        "trend_filter": [{"band": value} for value in (0.0, 0.01, 0.02)],
        "trend_probability_hybrid": [
            {"probability_spread": spread, "severe_minimum": minimum}
            for spread, minimum in ((0.07, -0.06), (0.10, -0.08), (0.14, -0.10))
        ],
        "support_pullback": [
            {"support_distance": support, "resistance_distance": resistance}
            for support, resistance in ((0.025, 0.02), (0.035, 0.025), (0.05, 0.035))
        ],
    }


def build_strategy_exposure(
    name: str,
    bars: pd.DataFrame,
    probabilities: pd.DataFrame,
    levels: pd.DataFrame,
    features: pd.DataFrame,
    minimum_risk: pd.Series,
    parameters: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Dispatch a named candidate without using any future observations."""
    params = dict(parameters or {})
    if name == "defensive_probability_overlay":
        return defensive_probability_overlay(probabilities, **params)
    if name == "trend_filter":
        return trend_filter(bars, probabilities.index, **params)
    if name == "trend_probability_hybrid":
        return trend_probability_hybrid(bars, probabilities, minimum_risk, **params)
    if name == "support_pullback":
        return support_pullback(bars, probabilities, levels, features, **params)
    raise ValueError(f"unknown strategy: {name}")


__all__ = [
    "STRATEGY_NAMES",
    "build_strategy_exposure",
    "defensive_probability_overlay",
    "strategy_parameter_grid",
    "support_pullback",
    "trend_filter",
    "trend_probability_hybrid",
]
