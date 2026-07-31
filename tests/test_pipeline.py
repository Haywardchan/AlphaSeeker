import numpy as np

from alpha_seeker.pipeline import run_analysis


def test_pipeline_smoke_uses_injected_data(bars):
    result = run_analysis(
        lookback_years=15,
        transaction_cost_bps=10,
        confidence_margin=0.05,
        bars=bars,
    )
    assert result.as_of == bars.index[-1]
    assert result.current_price == bars["Close"].iloc[-1]
    assert np.isclose(sum(result.probabilities.values()), 1)
    assert result.support.price < result.current_price < result.resistance.price
    assert {"price", "lower", "upper", "distance_pct"} <= set(
        result.support.__dataclass_fields__
    )
    assert result.validation.validation_rows > 0
    assert not result.backtest.equity.empty
    assert result.range_summary.minimum.median > 0
    assert np.isfinite(result.spread_summary.median_pct)
    assert result.spread_summary.median_pct > 0
    assert -1.0 <= result.emotion.composite_score <= 1.0
    assert result.base_signal.action in {"BUY", "HOLD", "SELL_REDUCE"}
    assert result.signal.final_action in {"BUY", "HOLD", "SELL_REDUCE"}
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
