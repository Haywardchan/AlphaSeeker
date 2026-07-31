"""Causal swing-pivot and KDE support/resistance zones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from .features import atr


@dataclass(frozen=True, slots=True)
class PriceZone:
    price: float
    lower: float
    upper: float
    distance_pct: float
    strength: float = 0.0
    source: str = "atr_fallback"


def _level_from_pivots(
    history: pd.DataFrame,
    *,
    pivot_order: int,
    atr_value: float,
    grid_size: int,
) -> tuple[float, float, float, float, str]:
    """Return support, resistance, strengths and method for supplied history."""
    close = float(history["Close"].iloc[-1])
    confirmed = history.iloc[:-pivot_order] if len(history) > pivot_order else history.iloc[:0]
    if len(confirmed) < pivot_order * 2 + 3:
        return close - 2 * atr_value, close + 2 * atr_value, 0.0, 0.0, "atr_fallback"

    prominence = max(float((confirmed["High"] - confirmed["Low"]).median()) * 0.25, 1e-9)
    high_positions, _ = find_peaks(
        confirmed["High"].to_numpy(dtype=float),
        distance=pivot_order,
        prominence=prominence,
    )
    low_positions, _ = find_peaks(
        -confirmed["Low"].to_numpy(dtype=float),
        distance=pivot_order,
        prominence=prominence,
    )
    pivots = np.concatenate(
        [
            confirmed["High"].to_numpy(dtype=float)[high_positions],
            confirmed["Low"].to_numpy(dtype=float)[low_positions],
        ]
    )
    if len(pivots) < 4 or np.ptp(pivots) <= np.finfo(float).eps:
        return close - 2 * atr_value, close + 2 * atr_value, 0.0, 0.0, "atr_fallback"

    try:
        kde = gaussian_kde(pivots)
        grid = np.linspace(float(pivots.min()), float(pivots.max()), grid_size)
        density = np.asarray(kde(grid), dtype=float)
        modes, _ = find_peaks(density, distance=max(2, grid_size // 40))
    except (ValueError, np.linalg.LinAlgError):
        modes = np.array([], dtype=int)
        grid = density = np.array([], dtype=float)

    if not len(modes):
        return close - 2 * atr_value, close + 2 * atr_value, 0.0, 0.0, "atr_fallback"
    candidates = grid[modes]
    strengths = density[modes] / max(float(density[modes].max()), np.finfo(float).eps)
    below = np.flatnonzero(candidates < close)
    above = np.flatnonzero(candidates > close)
    support = float(candidates[below[-1]]) if len(below) else close - 2 * atr_value
    resistance = float(candidates[above[0]]) if len(above) else close + 2 * atr_value
    support_strength = float(strengths[below[-1]]) if len(below) else 0.0
    resistance_strength = float(strengths[above[0]]) if len(above) else 0.0
    source = "kde" if len(below) and len(above) else "kde_atr_fallback"
    return support, resistance, support_strength, resistance_strength, source


def causal_level_frame(
    bars: pd.DataFrame,
    *,
    lookback: int = 252,
    atr_period: int = 14,
    pivot_order: int = 5,
    grid_size: int = 256,
    update_every: int = 5,
) -> pd.DataFrame:
    """Build point-in-time levels using confirmed trailing pivots only."""
    if min(lookback, atr_period, pivot_order, update_every) < 1 or grid_size < 32:
        raise ValueError("invalid level parameters")
    volatility = atr(bars, atr_period)
    output = pd.DataFrame(
        index=bars.index,
        columns=[
            "support",
            "resistance",
            "support_strength",
            "resistance_strength",
            "source",
        ],
    )
    minimum = max(atr_period + 1, pivot_order * 2 + 3)
    for position in range(minimum - 1, len(bars), update_every):
        start = max(0, position - lookback + 1)
        history = bars.iloc[start : position + 1]
        atr_value = float(volatility.iloc[position])
        if not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = max(
                float((history["High"] - history["Low"]).tail(atr_period).median()),
                float(history["Close"].iloc[-1]) * 0.01,
            )
        output.iloc[position] = _level_from_pivots(
            history,
            pivot_order=pivot_order,
            atr_value=atr_value,
            grid_size=grid_size,
        )
    output = output.ffill()
    close = bars["Close"].astype(float)
    fallback = volatility.fillna(close * 0.02).clip(lower=close * 0.001)
    output["support"] = pd.to_numeric(output["support"], errors="coerce")
    output["resistance"] = pd.to_numeric(output["resistance"], errors="coerce")
    output["support"] = output["support"].where(output["support"] < close, close - 2 * fallback)
    output["resistance"] = output["resistance"].where(
        output["resistance"] > close, close + 2 * fallback
    )
    output["support"] = output["support"].clip(lower=np.finfo(float).eps)
    width = fallback * 0.35
    output["support_lower"] = (output["support"] - width).clip(lower=np.finfo(float).eps)
    output["support_upper"] = output["support"] + width
    output["resistance_lower"] = output["resistance"] - width
    output["resistance_upper"] = output["resistance"] + width
    output["atr"] = volatility
    for column in ("support_strength", "resistance_strength"):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["source"] = output["source"].fillna("unavailable")
    return output


def estimate_levels(
    bars: pd.DataFrame, *, lookback: int = 252, atr_period: int = 14
) -> tuple[PriceZone, PriceZone]:
    frame = causal_level_frame(bars, lookback=lookback, atr_period=atr_period)
    row = frame.iloc[-1]
    close = float(bars["Close"].iloc[-1])
    support = PriceZone(
        float(row["support"]),
        float(row["support_lower"]),
        float(row["support_upper"]),
        float((row["support"] - close) / close),
        float(row["support_strength"]),
        str(row["source"]),
    )
    resistance = PriceZone(
        float(row["resistance"]),
        float(row["resistance_lower"]),
        float(row["resistance_upper"]),
        float((row["resistance"] - close) / close),
        float(row["resistance_strength"]),
        str(row["source"]),
    )
    return support, resistance


__all__ = ["PriceZone", "causal_level_frame", "estimate_levels"]
