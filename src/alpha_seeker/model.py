"""Leakage-aware temporal classification for first-touch outcomes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CONFIG

CLASS_LABELS: tuple[str, ...] = ("lower_first", "neither", "upper_first")


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.train
        yield self.validation


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    model_name: str
    log_loss: float
    brier_score: float
    validation_rows: int
    model_comparison: pd.DataFrame
    folds: int
    calibration_rows: int = 0
    calibration_temperature: float = 1.0


@dataclass(slots=True)
class ClassifierResult:
    model: TemporalClassifier
    diagnostics: ModelDiagnostics
    probabilities: pd.Series
    oos_probabilities: pd.DataFrame

    @property
    def validation(self) -> ModelDiagnostics:
        return self.diagnostics

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_proba_frame(features)


class PriorClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def fit(self, x: Any, y: Sequence[str]) -> PriorClassifier:
        counts = pd.Series(y).value_counts()
        total = len(y) + self.smoothing * len(CLASS_LABELS)
        self.classes_ = np.asarray(CLASS_LABELS, dtype=object)
        self.prior_ = np.asarray(
            [(counts.get(label, 0) + self.smoothing) / total for label in CLASS_LABELS]
        )
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        return np.tile(self.prior_, (len(x), 1))

    def predict(self, x: Any) -> np.ndarray:
        return np.repeat(self.classes_[self.prior_.argmax()], len(x))


class TemporalClassifier:
    def __init__(
        self,
        estimator: BaseEstimator,
        feature_columns: Sequence[str],
        temperature: float = 1.0,
    ) -> None:
        self.estimator = estimator
        self.feature_columns = tuple(feature_columns)
        self.temperature = float(temperature)
        self.classes_ = np.asarray(CLASS_LABELS, dtype=object)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        x = _prediction_features(features, self.feature_columns)
        return _temperature_scale(
            _aligned_probabilities(self.estimator, x), self.temperature
        )

    def predict_proba_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.predict_proba(features), index=features.index, columns=CLASS_LABELS
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def walk_forward_splits(
    n_samples: int,
    initial_train_size: int = 756,
    validation_size: int = 252,
    step_size: int = 252,
    horizon: int = 10,
    *,
    gap: int | None = None,
) -> list[TemporalSplit]:
    """Return expanding temporal folds separated by the label horizon."""
    if gap is not None:
        horizon = gap
    if n_samples <= 0 or horizon < 0:
        return []
    initial = max(8, min(int(initial_train_size), max(8, n_samples // 2)))
    validation = max(1, min(int(validation_size), max(1, n_samples // 4)))
    step = max(1, int(step_size))
    output: list[TemporalSplit] = []
    train_end = initial
    while train_end + horizon < n_samples:
        start = train_end + horizon
        end = min(start + validation, n_samples)
        output.append(TemporalSplit(np.arange(train_end), np.arange(start, end)))
        if end == n_samples:
            break
        train_end += step
    return output


def multiclass_brier_score(y_true: Sequence[str], probabilities: np.ndarray) -> float:
    encoded = pd.Categorical(y_true, categories=CLASS_LABELS).codes
    if np.any(encoded < 0):
        raise ValueError(f"labels must be one of {CLASS_LABELS}")
    observed = np.eye(len(CLASS_LABELS))[encoded]
    return float(np.mean(np.sum((observed - probabilities) ** 2, axis=1)))


def train_classifier(
    features: pd.DataFrame, labels: pd.Series, config: Any | None = None
) -> ClassifierResult:
    """Select a classifier using horizon-gapped OOS predictions, then refit."""
    cfg = config or DEFAULT_CONFIG
    x, y = _clean_training_data(features, labels)
    if len(x) < 12:
        raise ValueError("at least 12 labelled feature rows are required")
    splits = walk_forward_splits(
        len(x),
        int(cfg.initial_train_size),
        int(cfg.validation_size),
        int(cfg.step_size),
        int(cfg.horizon),
    )
    # Cap expensive model comparisons while retaining multiple chronological
    # expanding folds and a multi-year OOS strategy sample.
    splits = splits[-3:]
    candidates = _candidates(int(cfg.random_state))
    comparisons: list[dict[str, Any]] = []
    model_oos: dict[str, pd.DataFrame] = {}
    for name, candidate in candidates.items():
        pieces: list[pd.DataFrame] = []
        truths: list[pd.Series] = []
        for split in splits:
            train_y = y.iloc[split.train]
            estimator = candidate
            if name != "prior" and train_y.nunique() < 2:
                estimator = PriorClassifier()
            try:
                fitted = clone(estimator).fit(x.iloc[split.train], train_y)
            except ValueError:
                continue
            index = x.index[split.validation]
            pieces.append(
                pd.DataFrame(
                    _aligned_probabilities(fitted, x.iloc[split.validation]),
                    index=index,
                    columns=CLASS_LABELS,
                )
            )
            truths.append(y.iloc[split.validation])
        oos = (
            pd.concat(pieces).loc[lambda z: ~z.index.duplicated(keep="first")]
            if pieces
            else pd.DataFrame(columns=CLASS_LABELS)
        )
        model_oos[name] = oos
        truth = y.reindex(oos.index)
        if len(oos):
            ll = float(log_loss(truth, oos, labels=list(CLASS_LABELS)))
            brier = multiclass_brier_score(truth, oos.to_numpy())
        else:
            ll = brier = float("nan")
        comparisons.append(
            {"model": name, "log_loss": ll, "brier_score": brier, "validation_rows": len(oos)}
        )
    comparison = pd.DataFrame(comparisons)
    finite = comparison[np.isfinite(comparison["log_loss"])]
    selected = (
        str(finite.sort_values(["log_loss", "brier_score"]).iloc[0]["model"])
        if not finite.empty
        else "prior"
    )
    selected_oos = model_oos[selected]
    selected_truth = y.reindex(selected_oos.index)
    calibration_rows = max(0, min(126, len(selected_oos) // 4))
    temperature = _fit_temperature(
        selected_oos.iloc[-calibration_rows:].to_numpy(),
        selected_truth.iloc[-calibration_rows:],
    ) if calibration_rows else 1.0
    calibrated_oos = pd.DataFrame(
        _temperature_scale(selected_oos.to_numpy(), temperature),
        index=selected_oos.index,
        columns=CLASS_LABELS,
    )
    estimator = clone(candidates[selected]).fit(x, y)
    wrapped = TemporalClassifier(estimator, x.columns, temperature)
    latest = wrapped.predict_proba_frame(features.iloc[[-1]]).iloc[0]
    selected_row = comparison.loc[comparison["model"] == selected].iloc[0]
    calibrated_loss = (
        float(log_loss(selected_truth, calibrated_oos, labels=list(CLASS_LABELS)))
        if len(calibrated_oos)
        else float(selected_row["log_loss"])
    )
    calibrated_brier = (
        multiclass_brier_score(selected_truth, calibrated_oos.to_numpy())
        if len(calibrated_oos)
        else float(selected_row["brier_score"])
    )
    diagnostics = ModelDiagnostics(
        selected,
        calibrated_loss,
        calibrated_brier,
        int(selected_row["validation_rows"]),
        comparison.sort_values(["log_loss", "brier_score"], na_position="last").reset_index(
            drop=True
        ),
        len(splits),
        calibration_rows,
        temperature,
    )
    # Historical strategy simulations use the raw fold-held-out probabilities.
    # The late calibration block is valid for the latest forecast but would leak
    # future calibration outcomes backward if applied to earlier OOS rows.
    return ClassifierResult(wrapped, diagnostics, latest, selected_oos)


fit_first_touch_model = train_classifier


def _candidates(random_state: int) -> dict[str, BaseEstimator]:
    return {
        "prior": PriorClassifier(),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=2_000, class_weight="balanced", random_state=random_state
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingClassifier(
                max_iter=20,
                learning_rate=0.05,
                max_leaf_nodes=8,
                max_bins=64,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
    }


def _clean_training_data(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    labels = (
        pd.Series(labels, index=features.index)
        if not isinstance(labels, pd.Series)
        else labels
    )
    common = features.index.intersection(labels.index).sort_values()
    x = features.loc[common].select_dtypes(include=[np.number, "bool"]).astype(float)
    y = labels.loc[common]
    valid = y.isin(CLASS_LABELS) & ~x.isna().all(axis=1)
    x = x.loc[valid].replace([np.inf, -np.inf], np.nan)
    x = x.loc[:, ~x.isna().all(axis=0)]
    return x, y.loc[valid].astype(str)


def _prediction_features(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = set(columns).difference(features.columns)
    if missing:
        raise ValueError(f"missing model features: {sorted(missing)}")
    return features.loc[:, list(columns)].astype(float).replace([np.inf, -np.inf], np.nan)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    logits = np.log(clipped) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities: np.ndarray, labels: pd.Series) -> float:
    if len(probabilities) < 10:
        return 1.0
    candidates = np.geomspace(0.4, 4.0, 61)
    losses = [
        log_loss(
            labels,
            _temperature_scale(probabilities, temperature),
            labels=list(CLASS_LABELS),
        )
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def _aligned_probabilities(estimator: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(estimator.predict_proba(x), dtype=float)
    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "steps"):
        classes = estimator.steps[-1][1].classes_
    output = np.full((len(x), len(CLASS_LABELS)), 1e-12)
    for source, label in enumerate(classes):
        if str(label) in CLASS_LABELS:
            output[:, CLASS_LABELS.index(str(label))] = raw[:, source]
    return output / output.sum(axis=1, keepdims=True)


__all__ = [
    "CLASS_LABELS",
    "ClassifierResult",
    "ModelDiagnostics",
    "TemporalSplit",
    "fit_first_touch_model",
    "multiclass_brier_score",
    "train_classifier",
    "walk_forward_splits",
]
