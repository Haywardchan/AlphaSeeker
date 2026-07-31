"""Conditional quantile models for future path minima and maxima."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import make_pipeline

from .config import DEFAULT_CONFIG

REQUIRED_QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


@dataclass(frozen=True, slots=True)
class RangeDiagnostics:
    model_comparison: pd.DataFrame
    selected_minimum_model: str
    selected_maximum_model: str
    validation_rows: int
    folds: int


@dataclass(frozen=True, slots=True)
class RangeSideSummary:
    median: float
    frame: pd.DataFrame
    quantile_prices: pd.Series


@dataclass(frozen=True, slots=True)
class RangeSummary:
    minimum: RangeSideSummary
    maximum: RangeSideSummary


class EmpiricalQuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, quantile: float = 0.5):
        self.quantile = quantile

    def fit(self, x: Any, y: Sequence[float]) -> EmpiricalQuantileRegressor:
        self.value_ = float(np.nanquantile(np.asarray(y, dtype=float), self.quantile))
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.full(len(x), self.value_)


class RangeModel:
    """Monotone marginal range forecasts on the current price scale."""

    def __init__(
        self,
        current_price: float,
        quantiles: Sequence[float],
        minimum_returns: Sequence[float],
        maximum_returns: Sequence[float],
        diagnostics: RangeDiagnostics,
        minimum_estimators: dict[float, BaseEstimator],
        maximum_estimators: dict[float, BaseEstimator],
        feature_columns: Sequence[str],
        minimum_adjustments: Sequence[float] | None = None,
        maximum_adjustments: Sequence[float] | None = None,
    ) -> None:
        self.current_price = float(current_price)
        self.quantiles = np.asarray(quantiles, dtype=float)
        minimum = np.maximum.accumulate(np.asarray(minimum_returns, dtype=float))
        maximum = np.maximum.accumulate(np.asarray(maximum_returns, dtype=float))
        minimum = np.maximum(minimum, -0.999999)
        maximum = np.maximum(maximum, minimum)
        self.minimum_returns, self.maximum_returns = minimum, maximum
        self.minimum_estimators = minimum_estimators
        self.maximum_estimators = maximum_estimators
        self.feature_columns = tuple(feature_columns)
        self.minimum_adjustments = np.asarray(
            minimum_adjustments if minimum_adjustments is not None else np.zeros(len(quantiles))
        )
        self.maximum_adjustments = np.asarray(
            maximum_adjustments if maximum_adjustments is not None else np.zeros(len(quantiles))
        )
        self.diagnostics = diagnostics

    @property
    def minimum_prices(self) -> np.ndarray:
        return self.current_price * (1 + self.minimum_returns)

    @property
    def maximum_prices(self) -> np.ndarray:
        return self.current_price * (1 + self.maximum_returns)

    @property
    def minimum_cdf(self) -> np.ndarray:
        return self.quantiles.copy()

    @property
    def maximum_cdf(self) -> np.ndarray:
        return self.quantiles.copy()

    @property
    def summary(self) -> RangeSummary:
        return RangeSummary(
            _side_summary(self.quantiles, self.minimum_prices),
            _side_summary(self.quantiles, self.maximum_prices),
        )

    def minimum_below(self, price: float) -> float:
        return _cdf_at(price, self.minimum_prices, self.quantiles)

    def maximum_above(self, price: float) -> float:
        return 1 - _cdf_at(price, self.maximum_prices, self.quantiles)

    def predict(
        self, features: pd.DataFrame, current_price: float | None = None
    ) -> RangeModel:
        row = _prediction_row(features, self.feature_columns)
        minimum = [
            self.minimum_estimators[q].predict(row)[0] + adjustment
            for q, adjustment in zip(
                self.quantiles, self.minimum_adjustments, strict=True
            )
        ]
        maximum = [
            self.maximum_estimators[q].predict(row)[0] + adjustment
            for q, adjustment in zip(
                self.quantiles, self.maximum_adjustments, strict=True
            )
        ]
        return RangeModel(
            current_price or self.current_price,
            self.quantiles,
            minimum,
            maximum,
            self.diagnostics,
            self.minimum_estimators,
            self.maximum_estimators,
            self.feature_columns,
            self.minimum_adjustments,
            self.maximum_adjustments,
        )


def train_range_model(
    features: pd.DataFrame,
    future_min_return: pd.Series | pd.DataFrame,
    future_max_return: pd.Series | None = None,
    current_price: float | None = None,
    config: Any | None = None,
) -> RangeModel:
    if isinstance(future_min_return, pd.DataFrame):
        targets = future_min_return
        minimum = targets["future_min_return"]
        maximum = targets["future_max_return"]
    else:
        minimum, maximum = future_min_return, future_max_return
    if maximum is None:
        raise ValueError("future_max_return is required")
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price must be positive")
    cfg = config or DEFAULT_CONFIG
    x, y_min, y_max = _clean(features, minimum, maximum)
    if x.empty:
        raise ValueError("no usable range training rows")
    conditional = len(x) >= 150
    validation_rows = max(10, min(126, len(x) // 5))
    train_end = len(x) - validation_rows - int(cfg.horizon)
    comparisons: list[dict[str, Any]] = []
    min_selected, min_adjustments, minimum_models = _select_and_calibrate(
        "future_min_return",
        x,
        y_min,
        train_end,
        validation_rows,
        conditional,
        int(cfg.random_state),
        comparisons,
    )
    max_selected, max_adjustments, maximum_models = _select_and_calibrate(
        "future_max_return",
        x,
        y_max,
        train_end,
        validation_rows,
        conditional,
        int(cfg.random_state),
        comparisons,
    )
    latest = _prediction_row(features, x.columns)
    minimum_forecast = [
        minimum_models[q].predict(latest)[0] + adjustment
        for q, adjustment in zip(REQUIRED_QUANTILES, min_adjustments, strict=True)
    ]
    maximum_forecast = [
        maximum_models[q].predict(latest)[0] + adjustment
        for q, adjustment in zip(REQUIRED_QUANTILES, max_adjustments, strict=True)
    ]
    diagnostics = RangeDiagnostics(
        pd.DataFrame(comparisons),
        min_selected,
        max_selected,
        validation_rows if train_end >= 20 else 0,
        1 if train_end >= 20 else 0,
    )
    return RangeModel(
        current_price,
        REQUIRED_QUANTILES,
        minimum_forecast,
        maximum_forecast,
        diagnostics,
        minimum_models,
        maximum_models,
        x.columns,
        min_adjustments,
        max_adjustments,
    )


fit_range_model = train_range_model


def _select_and_calibrate(
    target_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    train_end: int,
    validation_rows: int,
    conditional_available: bool,
    random_state: int,
    comparisons: list[dict[str, Any]],
) -> tuple[str, np.ndarray, dict[float, BaseEstimator]]:
    if train_end < 20:
        comparisons.append(
            {
                "target": target_name,
                "model": "unconditional",
                "mean_pinball_loss": np.nan,
                "validation_rows": 0,
            }
        )
        models = {
            quantile: _fit_estimator(
                quantile, x, y, False, random_state
            )
            for quantile in REQUIRED_QUANTILES
        }
        return "unconditional", np.zeros(len(REQUIRED_QUANTILES)), models

    validation_start = len(x) - validation_rows
    train_x, train_y = x.iloc[:train_end], y.iloc[:train_end]
    validation_x, validation_y = x.iloc[validation_start:], y.iloc[validation_start:]
    candidates = ["unconditional"]
    if conditional_available:
        candidates.append("hist_gradient_boosting")
    scores: dict[str, float] = {}
    for name in candidates:
        median_model = _fit_estimator(
            0.50,
            train_x,
            train_y,
            name == "hist_gradient_boosting",
            random_state,
        )
        scores[name] = float(
            mean_pinball_loss(
                validation_y,
                median_model.predict(validation_x),
                alpha=0.50,
            )
        )
        comparisons.append(
            {
                "target": target_name,
                "model": name,
                "mean_pinball_loss": scores[name],
                "validation_rows": len(validation_y),
            }
        )
    selected = min(scores, key=scores.get)
    selected_models = {
        quantile: _fit_estimator(
            quantile,
            train_x,
            train_y,
            selected == "hist_gradient_boosting",
            random_state,
        )
        for quantile in REQUIRED_QUANTILES
    }
    adjustments = [
        float(
            np.quantile(
                validation_y.to_numpy() - model.predict(validation_x),
                quantile,
            )
        )
        for quantile, model in selected_models.items()
    ]
    return selected, np.asarray(adjustments), selected_models


def _fit_estimator(
    quantile: float,
    x: pd.DataFrame,
    y: pd.Series,
    conditional: bool,
    random_state: int,
) -> BaseEstimator:
    if conditional:
        model: BaseEstimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=25,
                max_leaf_nodes=8,
                max_bins=64,
                min_samples_leaf=20,
                early_stopping=True,
                random_state=random_state,
            ),
        )
    else:
        model = EmpiricalQuantileRegressor(quantile)
    return model.fit(x, y)


def _clean(
    features: pd.DataFrame, minimum: pd.Series, maximum: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    common = features.index.intersection(minimum.index).intersection(maximum.index).sort_values()
    x = features.loc[common].select_dtypes(include=[np.number, "bool"]).astype(float)
    y_min = pd.to_numeric(minimum.loc[common], errors="coerce")
    y_max = pd.to_numeric(maximum.loc[common], errors="coerce")
    valid = y_min.notna() & y_max.notna() & ~x.isna().all(axis=1)
    x = x.loc[valid].replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, ~x.isna().all(axis=0)]
    return x, y_min.loc[valid], y_max.loc[valid]


def _prediction_row(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = set(columns).difference(features.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    return features.loc[:, list(columns)].iloc[[-1]].replace([np.inf, -np.inf], np.nan)


def _side_summary(quantiles: np.ndarray, prices: np.ndarray) -> RangeSideSummary:
    lookup = dict(zip(quantiles, prices, strict=True))
    frame = pd.DataFrame(
        [
            {"interval": "50%", "lower": lookup[0.25], "upper": lookup[0.75]},
            {"interval": "80%", "lower": lookup[0.10], "upper": lookup[0.90]},
            {"interval": "95%", "lower": lookup[0.025], "upper": lookup[0.975]},
        ]
    )
    return RangeSideSummary(float(lookup[0.50]), frame, pd.Series(prices, index=quantiles))


def _cdf_at(value: float, values: np.ndarray, probabilities: np.ndarray) -> float:
    unique, inverse = np.unique(values, return_inverse=True)
    cumulative = np.asarray(
        [probabilities[inverse == index].max() for index in range(len(unique))]
    )
    return float(np.clip(np.interp(value, unique, cumulative, left=0, right=1), 0, 1))


__all__ = ["RangeModel", "RangeSummary", "fit_range_model", "train_range_model"]
