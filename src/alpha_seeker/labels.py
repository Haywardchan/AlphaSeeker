"""Forward targets built from point-in-time support and resistance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .levels import causal_level_frame


def future_extrema(bars: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Return next-horizon path minimum/maximum returns, excluding today."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    close = bars["Close"].to_numpy(dtype=float)
    low = bars["Low"].to_numpy(dtype=float)
    high = bars["High"].to_numpy(dtype=float)
    minimum = np.full(len(bars), np.nan)
    maximum = np.full(len(bars), np.nan)
    for index in range(len(bars) - horizon):
        future = slice(index + 1, index + horizon + 1)
        minimum[index] = np.round(low[future].min() / close[index] - 1, 12)
        maximum[index] = np.round(high[future].max() / close[index] - 1, 12)
    return pd.DataFrame(
        {"future_min_return": minimum, "future_max_return": maximum}, index=bars.index
    )


def first_touch_labels(
    bars: pd.DataFrame,
    levels: pd.DataFrame | None = None,
    *,
    horizon: int = 10,
) -> pd.Series:
    """Label which point-in-time level is touched first; ambiguous days are missing."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    level_frame = causal_level_frame(bars) if levels is None else levels
    support_column = "support" if "support" in level_frame else "support_price"
    resistance_column = (
        "resistance" if "resistance" in level_frame else "resistance_price"
    )
    low = bars["Low"].to_numpy(dtype=float)
    high = bars["High"].to_numpy(dtype=float)
    support = level_frame[support_column].reindex(bars.index).to_numpy(dtype=float)
    resistance = level_frame[resistance_column].reindex(bars.index).to_numpy(dtype=float)
    labels = np.full(len(bars), None, dtype=object)
    for index in range(len(bars) - horizon):
        if not np.isfinite(support[index]) or not np.isfinite(resistance[index]):
            continue
        outcome = "neither"
        for future in range(index + 1, index + horizon + 1):
            lower = low[future] <= support[index]
            upper = high[future] >= resistance[index]
            if lower and upper:
                outcome = None
                break
            if upper:
                outcome = "upper_first"
                break
            if lower:
                outcome = "lower_first"
                break
        labels[index] = outcome
    return pd.Series(labels, index=bars.index, name="first_touch", dtype="object")


def next_day_spread(bars: pd.DataFrame) -> pd.Series:
    """Return next-session intraday range as a fraction of today's close."""
    close = bars["Close"].to_numpy(dtype=float)
    spread = (bars["High"].shift(-1) - bars["Low"].shift(-1)).to_numpy(dtype=float)
    values = np.round(spread / close, 12)
    return pd.Series(values, index=bars.index, name="next_day_spread")


make_labels = first_touch_labels
make_range_targets = future_extrema

__all__ = [
    "first_touch_labels",
    "future_extrema",
    "make_labels",
    "make_range_targets",
    "next_day_spread",
]
