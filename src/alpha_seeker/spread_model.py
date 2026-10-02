"""Conditional quantile model for next-session intraday spread."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .config import DEFAULT_CONFIG
from .range_model import (
    REQUIRED_QUANTILES,
    _cdf_at,
    _prediction_row,
    _select_and_calibrate,
    _walk_forward_interval_coverage,
)


@dataclass(frozen=True, slots=True)
class SpreadDiagnostics:
    model_comparison: pd.DataFrame
    selected_model: str
    validation_rows: int
    folds: int
    coverage_80: float = float("nan")
    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None


@dataclass(frozen=True, slots=True)
class SpreadSummary:
    median_pct: float
    median_usd: float
    frame: pd.DataFrame
    quantile_spreads: pd.Series


class SpreadModel:
    """Monotone next-day spread forecasts as a fraction of the current close."""

    def __init__(
        self,
        current_price: float,
        quantiles: Sequence[float],
        spread_values: Sequence[float],
        diagnostics: SpreadDiagnostics,
        estimators: dict[float, BaseEstimator],
        feature_columns: Sequence[str],
        adjustments: Sequence[float] | None = None,
    ) -> None:
        self.current_price = float(current_price)
        self.quantiles = np.asarray(quantiles, dtype=float)
        spread = np.maximum.accumulate(np.asarray(spread_values, dtype=float))
        self.spread_values = np.maximum(spread, 0.0)
        self.estimators = estimators
        self.feature_columns = tuple(feature_columns)
        self.adjustments = np.asarray(
            adjustments if adjustments is not None else np.zeros(len(quantiles))
        )
        self.diagnostics = diagnostics

    @property
    def spread_usd(self) -> np.ndarray:
        return self.current_price * self.spread_values

    @property
    def spread_cdf(self) -> np.ndarray:
        return self.quantiles.copy()

    @property
    def summary(self) -> SpreadSummary:
        lookup = dict(zip(self.quantiles, self.spread_values, strict=True))
        usd_lookup = {quantile: value * self.current_price for quantile, value in lookup.items()}
        frame = pd.DataFrame(
            [
                {
                    "interval": "50%",
                    "lower_pct": lookup[0.25],
                    "upper_pct": lookup[0.75],
                    "lower_usd": usd_lookup[0.25],
                    "upper_usd": usd_lookup[0.75],
                },
                {
                    "interval": "80%",
                    "lower_pct": lookup[0.10],
                    "upper_pct": lookup[0.90],
                    "lower_usd": usd_lookup[0.10],
                    "upper_usd": usd_lookup[0.90],
                },
                {
                    "interval": "95%",
                    "lower_pct": lookup[0.025],
                    "upper_pct": lookup[0.975],
                    "lower_usd": usd_lookup[0.025],
                    "upper_usd": usd_lookup[0.975],
                },
            ]
        )
        return SpreadSummary(
            float(lookup[0.50]),
            float(usd_lookup[0.50]),
            frame,
            pd.Series(self.spread_values, index=self.quantiles),
        )

    def spread_at_or_above(self, threshold_pct: float) -> float:
        return 1 - _cdf_at(threshold_pct, self.spread_values, self.quantiles)

    def predict(
        self, features: pd.DataFrame, current_price: float | None = None
    ) -> SpreadModel:
        row = _prediction_row(features, self.feature_columns)
        forecast = [
            self.estimators[q].predict(row)[0] + adjustment
            for q, adjustment in zip(self.quantiles, self.adjustments, strict=True)
        ]
        return SpreadModel(
            current_price or self.current_price,
            self.quantiles,
            forecast,
            self.diagnostics,
            self.estimators,
            self.feature_columns,
            self.adjustments,
        )


def train_spread_model(
    features: pd.DataFrame,
    spread_target: pd.Series,
    current_price: float | None = None,
    config: Any | None = None,
) -> SpreadModel:
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price must be positive")
    cfg = config or DEFAULT_CONFIG
    x, y = _clean_spread(features, spread_target)
    if x.empty:
        raise ValueError("no usable spread training rows")
    conditional = len(x) >= 150
    validation_rows = max(10, min(126, len(x) // 5))
    train_end = len(x) - validation_rows - 1
    comparisons: list[dict[str, Any]] = []
    selected, adjustments, models = _select_and_calibrate(
        "next_day_spread",
        x,
        y,
        train_end,
        validation_rows,
        conditional,
        int(cfg.random_state),
        comparisons,
    )
    latest = _prediction_row(features, x.columns)
    forecast = [
        models[q].predict(latest)[0] + adjustment
        for q, adjustment in zip(REQUIRED_QUANTILES, adjustments, strict=True)
    ]
    coverage, folds, validation_start, validation_end = (
        _walk_forward_interval_coverage(
            x,
            y,
            validation_rows,
            1,
            int(cfg.random_state),
        )
    )
    diagnostics = SpreadDiagnostics(
        model_comparison=pd.DataFrame(comparisons),
        selected_model=selected,
        validation_rows=validation_rows if train_end >= 20 else 0,
        folds=folds,
        coverage_80=coverage,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    return SpreadModel(
        current_price,
        REQUIRED_QUANTILES,
        forecast,
        diagnostics,
        models,
        x.columns,
        adjustments,
    )


def _clean_spread(
    features: pd.DataFrame, spread: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    common = features.index.intersection(spread.index).sort_values()
    x = features.loc[common].select_dtypes(include=[np.number, "bool"]).astype(float)
    y = pd.to_numeric(spread.loc[common], errors="coerce")
    valid = y.notna() & ~x.isna().all(axis=1)
    x = x.loc[valid].replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, ~x.isna().all(axis=0)]
    return x, y.loc[valid]


__all__ = ["SpreadDiagnostics", "SpreadModel", "SpreadSummary", "train_spread_model"]
