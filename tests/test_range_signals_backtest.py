import numpy as np
import pandas as pd

from alpha_seeker.backtest import run_backtest, run_exposure_backtest
from alpha_seeker.range_model import train_range_model
from alpha_seeker.signals import Guidance, generate_signal


def test_range_quantiles_monotone_and_cdf_bounded():
    index = pd.bdate_range("2023-01-01", periods=80)
    features = pd.DataFrame({"x": np.sin(np.arange(80))}, index=index)
    minimum = pd.Series(np.linspace(-0.1, -0.01, 80), index=index)
    maximum = pd.Series(np.linspace(0.01, 0.1, 80), index=index)
    model = train_range_model(features, minimum, maximum, current_price=100)
    assert np.all(np.diff(model.minimum_prices) >= 0)
    assert np.all(np.diff(model.maximum_prices) >= 0)
    assert 0 <= model.minimum_below(95) <= 1
    assert 0 <= model.maximum_above(105) <= 1
    assert model.minimum_below(90) <= model.minimum_below(100)
    assert model.maximum_above(100) >= model.maximum_above(110)
    assert list(model.summary.minimum.frame["interval"]) == ["50%", "80%", "95%"]
    assert model.diagnostics.validation_rows > 0
    assert set(model.diagnostics.model_comparison["target"]) == {
        "future_min_return",
        "future_max_return",
    }


def test_signal_rules_are_symmetric():
    buy = generate_signal(
        {"upper_first": 0.6, "lower_first": 0.2, "neither": 0.2},
        100,
        90,
        110,
        confidence_margin=0.1,
    )
    sell = generate_signal(
        {"upper_first": 0.2, "lower_first": 0.6, "neither": 0.2},
        100,
        90,
        110,
        confidence_margin=0.1,
    )
    assert buy.action == Guidance.BUY
    assert sell.action == Guidance.SELL_REDUCE


def test_backtest_executes_close_signal_at_next_open():
    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {
            "Open": [100, 110, 120, 130],
            "High": [101, 111, 121, 131],
            "Low": [99, 109, 119, 129],
            "Close": [100, 110, 120, 130],
        },
        index=index,
    )
    signals = pd.Series(["BUY", "HOLD", "SELL_REDUCE", "HOLD"], index=index)
    result = run_backtest(bars, signals)
    assert result.equity.index[0] == index[1]
    assert result.equity.loc[index[1], "Position"] > 0.99
    assert result.equity.loc[index[3], "Position"] == 0
    assert np.isclose(result.metrics["total_return"], 130 / 110 - 1)
    assert np.isclose(result.equity.iloc[0]["BuyHold"], result.equity.iloc[0]["Strategy"])


def test_backtest_allows_tiny_adjusted_ohlc_rounding_difference():
    index = pd.bdate_range("2024-01-01", periods=4)
    bars = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [100.0 - 1e-12, 102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 101.0, 102.0, 103.0],
        },
        index=index,
    )
    signals = pd.Series(["HOLD"] * 4, index=index)
    result = run_backtest(bars, signals)
    assert not result.equity.empty


def test_fractional_exposure_is_next_open_and_costed():
    index = pd.bdate_range("2024-01-01", periods=5)
    bars = pd.DataFrame(
        {
            "Open": [100, 100, 100, 100, 100],
            "High": [101] * 5,
            "Low": [99] * 5,
            "Close": [100] * 5,
        },
        index=index,
    )
    targets = pd.Series([0.5, 1.0, 0.0], index=index[:3])
    result = run_exposure_backtest(bars, targets, transaction_cost_bps=10)
    assert result.equity.index[0] == index[1]
    assert np.isclose(result.equity.iloc[0]["Position"], 0.5, atol=0.002)
    assert result.metrics["total_return"] < 0
    assert result.metrics["turnover"] > 1
