import numpy as np
import pandas as pd

from alpha_seeker.config import AlphaConfig
from alpha_seeker.features import make_features
from alpha_seeker.labels import first_touch_labels, future_extrema, next_day_spread
from alpha_seeker.levels import causal_level_frame
from alpha_seeker.model import train_classifier, walk_forward_splits


def test_labels_exclude_same_day_dual_touch():
    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {
            "Open": [100] * 4,
            "High": [101, 111, 102, 102],
            "Low": [99, 89, 98, 98],
            "Close": [100] * 4,
            "Volume": [1] * 4,
        },
        index=index,
    )
    levels = pd.DataFrame({"support": [90] * 4, "resistance": [110] * 4}, index=index)
    labels = first_touch_labels(bars, levels, horizon=2)
    assert pd.isna(labels.iloc[0])


def test_future_extrema_excludes_today_and_has_nan_tail():
    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {
            "Open": [10, 10, 10, 10],
            "High": [99, 12, 13, 14],
            "Low": [1, 8, 7, 6],
            "Close": [10, 10, 10, 10],
            "Volume": [1] * 4,
        },
        index=index,
    )
    result = future_extrema(bars, horizon=2)
    assert result.iloc[0]["future_min_return"] == -0.3
    assert result.iloc[0]["future_max_return"] == 0.3
    assert result.tail(2).isna().all().all()


def test_next_day_spread_uses_next_bar_only_and_has_nan_tail():
    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {
            "Open": [10, 10, 10, 10],
            "High": [11, 12, 13, 14],
            "Low": [9, 8, 7, 6],
            "Close": [10, 10, 10, 10],
            "Volume": [1] * 4,
        },
        index=index,
    )
    result = next_day_spread(bars)
    assert result.iloc[0] == 0.4
    assert result.iloc[1] == 0.6
    assert pd.isna(result.iloc[-1])


def test_temporal_split_has_horizon_gap():
    splits = walk_forward_splits(100, 40, 20, 20, 10)
    assert splits
    assert all(split.validation.min() - split.train.max() == 11 for split in splits)


def test_classifier_probabilities_sum_and_oos_is_later(bars):
    features = make_features(bars, windows=(5, 10, 20))
    levels = causal_level_frame(bars, lookback=40)
    labels = first_touch_labels(bars, levels, horizon=5)
    config = AlphaConfig(
        horizon=5, initial_train_size=100, validation_size=40, step_size=40
    )
    result = train_classifier(features, labels, config)
    assert np.isclose(result.probabilities.sum(), 1)
    assert np.allclose(result.oos_probabilities.sum(axis=1), 1)
    assert result.oos_probabilities.index.min() > features.index[99]
    assert result.validation.calibration_rows > 0
    assert np.isfinite(result.validation.calibration_temperature)
    assert result.validation.model_comparison["model"].nunique() == 3
