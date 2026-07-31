"""Causal pandas/numpy technical features for daily bars."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import validate_ohlcv


def true_range(data: pd.DataFrame) -> pd.Series:
    """Compute Wilder true range from current and previous bars only."""
    _require(data, "High", "Low", "Close")
    previous = data["Close"].shift(1)
    return pd.concat(
        (data["High"] - data["Low"], (data["High"] - previous).abs(),
         (data["Low"] - previous).abs()), axis=1
    ).max(axis=1).rename("true_range")


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute causal Wilder average true range."""
    _period(period)
    return true_range(data).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean().rename(f"atr_{period}")


average_true_range = atr


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder RSI, with flat windows represented by 50."""
    _period(period)
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    result = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return result.mask((loss == 0) & (gain > 0), 100).mask(
        (loss == 0) & (gain == 0), 50
    ).rename(f"rsi_{period}")


def build_feature_frame(
    data: pd.DataFrame,
    *,
    atr_period: int = 14,
    short_window: int = 10,
    long_window: int = 20,
    rsi_period: int = 14,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Build common predictors; warm-up rows intentionally remain missing."""
    _require(data, "Open", "High", "Low", "Close", "Volume")
    for value in (atr_period, short_window, long_window, rsi_period, volatility_window):
        _period(value)
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")
    close, volume = data["Close"].astype(float), data["Volume"].astype(float)
    log_return = np.log(close).diff()
    short_sma = close.rolling(short_window, min_periods=short_window).mean()
    long_sma = close.rolling(long_window, min_periods=long_window).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    std = close.rolling(long_window, min_periods=long_window).std(ddof=0)
    volume_roll = volume.rolling(long_window, min_periods=long_window)
    result = pd.DataFrame(index=data.index)
    result["return_1d"] = close.pct_change(fill_method=None)
    result["log_return_1d"] = log_return
    result[f"return_{short_window}d"] = close.pct_change(
        short_window, fill_method=None
    )
    result[f"sma_ratio_{short_window}"] = close / short_sma - 1
    result[f"sma_ratio_{long_window}"] = close / long_sma - 1
    result["sma_cross"] = short_sma / long_sma - 1
    result[f"rsi_{rsi_period}"] = rsi(close, rsi_period)
    result["macd"] = macd
    result["macd_signal"] = signal
    result["macd_histogram"] = macd - signal
    result[f"atr_{atr_period}"] = atr(data, atr_period)
    result[f"atr_pct_{atr_period}"] = result[f"atr_{atr_period}"] / close
    result[f"volatility_{volatility_window}"] = log_return.rolling(
        volatility_window, min_periods=volatility_window
    ).std(ddof=0)
    result[f"bollinger_position_{long_window}"] = (
        close - long_sma
    ) / (2 * std.replace(0, np.nan))
    result[f"volume_zscore_{long_window}"] = (
        volume - volume_roll.mean()
    ) / volume_roll.std(ddof=0).replace(0, np.nan)
    result["close_location"] = (close - data["Low"]) / (
        data["High"] - data["Low"]
    ).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


technical_features = build_feature_frame


def make_features(
    data: pd.DataFrame,
    *,
    windows: Iterable[int] = (5, 10, 20, 50, 100, 200),
    validate: bool = True,
) -> pd.DataFrame:
    """Build a broader causal feature set used by the modelling pipeline."""
    bars = validate_ohlcv(data) if validate else data.copy()
    close, volume = bars["Close"], bars["Volume"]
    result = build_feature_frame(bars)
    result["overnight_return"] = bars["Open"] / close.shift(1) - 1
    result["intraday_return"] = close / bars["Open"] - 1
    result["range_pct"] = (bars["High"] - bars["Low"]) / close
    valid_windows = sorted({int(value) for value in windows if int(value) > 1})
    if not valid_windows:
        raise ValueError("windows must contain an integer greater than one")
    daily_return = close.pct_change(fill_method=None)
    for window in valid_windows:
        rolling_close = close.rolling(window, min_periods=window)
        rolling_volume = volume.rolling(window, min_periods=window)
        result[f"close_sma_{window}"] = close / rolling_close.mean() - 1
        result[f"momentum_{window}"] = close.pct_change(window, fill_method=None)
        result[f"volatility_{window}"] = daily_return.rolling(
            window, min_periods=window
        ).std(ddof=0) * np.sqrt(252)
        result[f"volume_ratio_{window}"] = volume / rolling_volume.mean() - 1
    result["rsi_14"] = rsi(close, 14) / 100
    result["atr_14_pct"] = atr(bars, 14) / close
    return result.replace([np.inf, -np.inf], np.nan).astype(float)


add_technical_features = make_features
compute_features = make_features


def _require(data: pd.DataFrame, *columns: str) -> None:
    missing = [name for name in columns if name not in data]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


def _period(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("periods must be positive integers")

