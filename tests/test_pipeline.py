import numpy as np
import pytest

from alpha_seeker.pipeline import run_analysis


@pytest.mark.parametrize("symbol", ["BRK.B", "VTI"])
def test_pipeline_smoke_uses_injected_data(bars, symbol):
    result = run_analysis(
        symbol=symbol,
        lookback_years=15,
        transaction_cost_bps=10,
        confidence_margin=0.05,
        bars=bars,
    )
    assert result.as_of == bars.index[-1]
    assert result.symbol == symbol.upper().replace(".", "-")
    assert result.loaded_at >= result.as_of
    assert not result.benchmark_bars.empty
    assert result.current_price == bars["Close"].iloc[-1]
    assert np.isclose(sum(result.probabilities.values()), 1)
    assert result.support.price < result.current_price < result.resistance.price
    assert {"price", "lower", "upper", "distance_pct"} <= set(
        result.support.__dataclass_fields__
    )
    assert result.validation.validation_rows > 0
    assert result.validation.validation_start < result.validation.validation_end
    assert 0 <= result.ambiguous_label_rate <= 1
    assert not result.backtest.equity.empty
    assert result.range_summary.minimum.median > 0
    assert result.range_model.diagnostics.folds > 1
    assert 0 <= result.range_model.diagnostics.minimum_80_coverage <= 1
    assert np.isfinite(result.spread_summary.median_pct)
    assert result.spread_summary.median_pct > 0
    assert result.spread_model.diagnostics.folds > 1
    assert 0 <= result.spread_model.diagnostics.coverage_80 <= 1
    assert -1.0 <= result.emotion.composite_score <= 1.0
    assert result.base_signal.action in {"BUY", "HOLD", "SELL_REDUCE"}
    assert result.signal.final_action in {"BUY", "HOLD", "SELL_REDUCE"}
    advice = result.historical_advice
    assert not advice.empty
    assert advice.index.is_monotonic_increasing
    assert advice.index.is_unique
    assert advice.index.max() <= result.bars.index[-11]
    assert advice[["actual_outcome", "future_return"]].notna().all().all()
    assert set(advice["action"]) <= {"BUY", "HOLD", "SELL_REDUCE"}
    expected = advice["action"].map(
        {
            "BUY": "upper_first",
            "HOLD": "neither",
            "SELL_REDUCE": "lower_first",
        }
    )
    assert advice["correct"].equals(expected.eq(advice["actual_outcome"]))
    assert result.backtest.equity.index[0] > advice.index[0]
    lab = result.strategy_lab
    assert lab.development_end < lab.holdout_start
    assert set(lab.leaderboard["strategy"]) == {
        "defensive_probability_overlay",
        "trend_filter",
        "trend_probability_hybrid",
        "support_pullback",
    }
    assert lab.exposures.index.min() == lab.holdout_start
    assert lab.winner == "buy_and_hold" or bool(
        lab.leaderboard.set_index("strategy").at[lab.winner, "eligible"]
    )
    for backtest in lab.backtests.values():
        assert backtest.equity.index[0] > lab.holdout_start
        assert backtest.metrics["max_drawdown"] <= 0
