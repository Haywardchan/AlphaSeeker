import numpy as np
import pandas as pd

from alpha_seeker.spread_model import train_spread_model


def test_spread_quantiles_monotone_and_cdf_bounded():
    index = pd.bdate_range("2023-01-01", periods=80)
    features = pd.DataFrame({"x": np.sin(np.arange(80))}, index=index)
    spread = pd.Series(np.linspace(0.01, 0.05, 80), index=index)
    model = train_spread_model(features, spread, current_price=100)
    assert np.all(np.diff(model.spread_values) >= 0)
    assert np.all(model.spread_values >= 0)
    assert 0 <= model.spread_at_or_above(0.02) <= 1
    assert model.spread_at_or_above(0.02) >= model.spread_at_or_above(0.04)
    assert list(model.summary.frame["interval"]) == ["50%", "80%", "95%"]
    assert model.summary.median_usd == model.summary.median_pct * 100
    assert model.diagnostics.validation_rows > 0
    assert set(model.diagnostics.model_comparison["target"]) == {"next_day_spread"}
