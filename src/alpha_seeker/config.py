"""Validated configuration for supported daily analytics instruments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SUPPORTED_SYMBOLS: tuple[str, ...] = ("BRK-B", "VTI")


@dataclass(frozen=True, slots=True)
class InstrumentProfile:
    """Instrument-specific assumptions layered over shared model settings."""

    display_name: str
    level_lookback: int
    news_weight_multiplier: float
    emotion_weights: tuple[tuple[str, float], ...]


INSTRUMENT_PROFILES: dict[str, InstrumentProfile] = {
    "BRK-B": InstrumentProfile(
        "Berkshire Hathaway Class B",
        756,
        1.0,
        (
            ("vix_zscore", 0.20),
            ("vix_term_proxy", 0.10),
            ("spy_trend", 0.20),
            ("spy_rsi", 0.10),
            ("spy_volume_z", 0.05),
            ("stock_rsi", 0.15),
            ("stock_volume_z", 0.10),
            ("trin_proxy", 0.10),
        ),
    ),
    "VTI": InstrumentProfile(
        "Vanguard Total Stock Market ETF",
        504,
        0.625,
        (
            ("vix_zscore", 0.25),
            ("vix_term_proxy", 0.10),
            ("spy_trend", 0.10),
            ("spy_rsi", 0.05),
            ("spy_volume_z", 0.025),
            ("stock_rsi", 0.25),
            ("stock_volume_z", 0.125),
            ("trin_proxy", 0.10),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class AlphaConfig:
    """Settings shared by data, feature, level, and modelling components."""

    symbol: str = "BRK-B"
    market_proxy: str = "SPY"
    fear_gauge: str = "^VIX"
    period: str = "max"
    interval: str = "1d"
    horizon: int = 10
    upper_return: float = 0.04
    lower_return: float = -0.04
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ALPHA_DATA_DIR", "data"))
    )
    cache_ttl_seconds: int | None = 86_400
    atr_window: int = 14
    pivot_order: int = 5
    level_lookback: int = 504
    kde_grid_size: int = 256
    kde_bandwidth: float | Literal["scott", "silverman"] = "scott"
    initial_train_size: int = 756
    validation_size: int = 504
    step_size: int = 504
    probability_threshold: float = 0.45
    min_expected_return: float = 0.0
    transaction_cost_bps: float = 5.0
    random_state: int = 42
    quantiles: tuple[float, ...] = field(
        default_factory=lambda: (0.1, 0.25, 0.5, 0.75, 0.9)
    )
    emotion_weight: float = 0.6
    news_weight: float = 0.4
    emotion_scale: float = 0.0075
    news_impact_cap: float = 0.015
    news_half_life_hours: float = 36.0
    news_max_articles: int = 15
    news_cache_ttl_seconds: int = 3600
    vix_stress_zscore: float = 2.0
    fusion_support_factor: float = 0.75
    fusion_flip_cost_multiple: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper().replace(".", "-"))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.symbol not in SUPPORTED_SYMBOLS:
            supported = ", ".join(SUPPORTED_SYMBOLS)
            raise ValueError(f"symbol must be one of: {supported}")
        if self.interval != "1d":
            raise ValueError("only daily ('1d') data is supported")
        for name in (
            "horizon", "atr_window", "pivot_order", "level_lookback",
            "kde_grid_size", "initial_train_size", "validation_size", "step_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.kde_grid_size < 32:
            raise ValueError("kde_grid_size must be at least 32")
        if not self.lower_return < 0 < self.upper_return:
            raise ValueError("lower_return must be negative and upper_return positive")
        if self.cache_ttl_seconds is not None and self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative or None")
        if not 0 <= self.probability_threshold <= 1:
            raise ValueError("probability_threshold must be in [0, 1]")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and increasing")
        if any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles must lie strictly between zero and one")
        if self.news_max_articles <= 0:
            raise ValueError("news_max_articles must be positive")
        if self.news_cache_ttl_seconds < 0:
            raise ValueError("news_cache_ttl_seconds must be non-negative")
        if self.news_half_life_hours <= 0:
            raise ValueError("news_half_life_hours must be positive")
        if self.news_impact_cap <= 0:
            raise ValueError("news_impact_cap must be positive")
        if self.emotion_scale <= 0:
            raise ValueError("emotion_scale must be positive")
        if self.emotion_weight < 0 or self.news_weight < 0:
            raise ValueError("emotion_weight and news_weight must be non-negative")
        if self.vix_stress_zscore <= 0:
            raise ValueError("vix_stress_zscore must be positive")
        if self.fusion_support_factor <= 0 or self.fusion_flip_cost_multiple <= 0:
            raise ValueError("fusion factors must be positive")

    @property
    def ticker(self) -> str:
        """Ticker spelling accepted by Yahoo Finance."""
        return self.symbol

    @property
    def profile(self) -> InstrumentProfile:
        """Return assumptions tailored to the selected instrument."""
        return INSTRUMENT_PROFILES[self.symbol]

    @property
    def effective_news_weight(self) -> float:
        return self.news_weight * self.profile.news_weight_multiplier

    @property
    def atr_period(self) -> int:
        """Compatibility name for the ATR window."""
        return self.atr_window

    @property
    def peak_distance(self) -> int:
        """Compatibility name for the pivot order."""
        return self.pivot_order


AnalyticsConfig = AlphaConfig
DEFAULT_CONFIG = AlphaConfig()

