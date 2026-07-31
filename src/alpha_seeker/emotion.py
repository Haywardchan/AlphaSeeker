"""Proxy market-emotion indicators from VIX, SPY, and BRK.B."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .config import AlphaConfig
from .features import rsi
from .market_data import MarketDataBundle


class EmotionRegime(StrEnum):
    FEAR = "Fear"
    NEUTRAL = "Neutral"
    GREED = "Greed"


@dataclass(frozen=True, slots=True)
class EmotionSnapshot:
    composite_score: float
    regime: EmotionRegime
    components: dict[str, float]
    trin_proxy: float
    as_of: pd.Timestamp
    history: pd.DataFrame


DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "vix_zscore": 0.20,
    "vix_term_proxy": 0.10,
    "spy_trend": 0.20,
    "spy_rsi": 0.10,
    "spy_volume_z": 0.05,
    "stock_rsi": 0.15,
    "stock_volume_z": 0.10,
    "trin_proxy": 0.10,
}


def compute_emotion(
    bundle: MarketDataBundle,
    config: AlphaConfig | None = None,
    *,
    history_days: int = 90,
) -> EmotionSnapshot:
    """Return the latest proxy emotion snapshot and recent history."""
    aligned = _align_bundle(bundle)
    latest = _latest_components(aligned)
    normalized = {name: _normalize_component(name, value) for name, value in latest.items()}
    composite = _weighted_composite(normalized, DEFAULT_COMPONENT_WEIGHTS)
    regime = _regime_from_score(composite)
    history = _emotion_history(aligned, history_days, DEFAULT_COMPONENT_WEIGHTS)
    return EmotionSnapshot(
        composite_score=composite,
        regime=regime,
        components={**latest, **{f"{key}_score": value for key, value in normalized.items()}},
        trin_proxy=float(latest["trin_proxy"]),
        as_of=pd.Timestamp(aligned.brkb.index[-1]),
        history=history,
    )


def _align_bundle(bundle: MarketDataBundle) -> MarketDataBundle:
    common = bundle.brkb.index.intersection(bundle.spy.index).intersection(bundle.vix.index)
    if len(common) < 30:
        raise ValueError("insufficient overlapping market history for emotion analysis")
    common = common.sort_values()
    return MarketDataBundle(
        bundle.brkb.loc[common],
        bundle.spy.loc[common],
        bundle.vix.loc[common],
    )


def _latest_components(bundle: MarketDataBundle) -> dict[str, float]:
    frame = _component_frame(bundle)
    latest = frame.iloc[-1]
    return {name: float(latest[name]) for name in frame.columns}


def _component_frame(bundle: MarketDataBundle) -> pd.DataFrame:
    brkb, spy, vix = bundle.brkb, bundle.spy, bundle.vix
    vix_close = vix["Close"].astype(float)
    vix_mean = vix_close.rolling(252, min_periods=60).mean()
    vix_std = vix_close.rolling(252, min_periods=60).std(ddof=0).replace(0, np.nan)
    vix_zscore = (vix_close - vix_mean) / vix_std
    vix_sma = vix_close.rolling(20, min_periods=10).mean()
    vix_term_proxy = vix_close / vix_sma - 1.0
    spy_close = spy["Close"].astype(float)
    spy_trend = spy_close / spy_close.rolling(200, min_periods=100).mean() - 1.0
    spy_rsi = rsi(spy_close, 14)
    spy_volume_z = _volume_zscore(spy["Volume"])
    stock_rsi = rsi(brkb["Close"].astype(float), 14)
    stock_volume_z = _volume_zscore(brkb["Volume"])
    trin_proxy = _trin_proxy(spy)
    return pd.DataFrame(
        {
            "vix_zscore": vix_zscore,
            "vix_term_proxy": vix_term_proxy,
            "spy_trend": spy_trend,
            "spy_rsi": spy_rsi,
            "spy_volume_z": spy_volume_z,
            "stock_rsi": stock_rsi,
            "stock_volume_z": stock_volume_z,
            "trin_proxy": trin_proxy,
        },
        index=brkb.index,
    ).replace([np.inf, -np.inf], np.nan)


def _emotion_history(
    bundle: MarketDataBundle,
    history_days: int,
    weights: dict[str, float],
) -> pd.DataFrame:
    frame = _component_frame(bundle).tail(history_days)
    scores = frame.apply(
        lambda row: _weighted_composite(
            {name: _normalize_component(name, float(row[name])) for name in frame.columns},
            weights,
        ),
        axis=1,
    )
    return pd.DataFrame(
        {
            "composite_score": scores,
            "vix_close": bundle.vix["Close"].astype(float).reindex(scores.index),
        },
        index=scores.index,
    )


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    rolling = volume.astype(float).rolling(window, min_periods=window)
    return (volume.astype(float) - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)


def _trin_proxy(spy: pd.DataFrame, window: int = 5) -> pd.Series:
    close = spy["Close"].astype(float)
    volume = spy["Volume"].astype(float)
    up = close.diff() > 0
    down = close.diff() < 0
    issue_ratio = up.rolling(window, min_periods=window).mean() / down.rolling(
        window, min_periods=window
    ).mean().replace(0, np.nan)
    up_volume = volume.where(up, 0.0).rolling(window, min_periods=window).sum()
    down_volume = volume.where(down, 0.0).rolling(window, min_periods=window).sum()
    volume_ratio = up_volume / down_volume.replace(0, np.nan)
    return (issue_ratio / volume_ratio).clip(0.2, 5.0).fillna(1.0)


def _normalize_component(name: str, value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if name == "vix_zscore":
        return float(np.clip(-value / 2.5, -1.0, 1.0))
    if name == "vix_term_proxy":
        return float(np.clip(-value / 0.25, -1.0, 1.0))
    if name == "spy_trend":
        return float(np.clip(value / 0.08, -1.0, 1.0))
    if name in {"spy_rsi", "stock_rsi"}:
        return float(np.clip((value - 50.0) / 25.0, -1.0, 1.0))
    if name in {"spy_volume_z", "stock_volume_z"}:
        return float(np.clip(value / 2.0, -1.0, 1.0))
    if name == "trin_proxy":
        return float(np.clip((1.0 - value) / 0.5, -1.0, 1.0))
    return float(np.clip(value, -1.0, 1.0))


def _weighted_composite(normalized: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    score = sum(normalized.get(name, 0.0) * weight for name, weight in weights.items()) / total
    return float(np.clip(score, -1.0, 1.0))


def _regime_from_score(score: float) -> EmotionRegime:
    if score <= -0.25:
        return EmotionRegime.FEAR
    if score >= 0.25:
        return EmotionRegime.GREED
    return EmotionRegime.NEUTRAL


__all__ = ["EmotionRegime", "EmotionSnapshot", "compute_emotion"]
