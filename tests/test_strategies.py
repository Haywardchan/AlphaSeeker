import numpy as np
import pandas as pd

from alpha_seeker.features import make_features
from alpha_seeker.levels import causal_level_frame
from alpha_seeker.strategies import (
    STRATEGY_NAMES,
    build_strategy_exposure,
    strategy_parameter_grid,
)


def test_all_strategy_candidates_are_bounded_and_causal(bars):
    features = make_features(bars)
    levels = causal_level_frame(bars, lookback=60)
    dates = bars.index[210:280]
    probabilities = pd.DataFrame(
        {
            "lower_first": 0.3 + 0.1 * np.sin(np.arange(len(dates)) / 5),
            "neither": 0.3,
            "upper_first": 0.4 - 0.1 * np.sin(np.arange(len(dates)) / 5),
        },
        index=dates,
    )
    risk = pd.Series(-0.05, index=bars.index)
    grids = strategy_parameter_grid()
    for name in STRATEGY_NAMES:
        exposure = build_strategy_exposure(
            name,
            bars,
            probabilities,
            levels,
            features,
            risk,
            grids[name][0],
        )
        assert exposure.index.equals(dates)
        assert exposure.between(0, 1).all()

