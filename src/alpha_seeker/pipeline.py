"""End-to-end app-facing BRK-B analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, run_exposure_backtest
from .config import AlphaConfig
from .data import validate_ohlcv
from .emotion import EmotionSnapshot, compute_emotion
from .features import make_features
from .fusion import FusedRecommendation, fuse_guidance
from .labels import first_touch_labels, future_extrema, next_day_spread
from .levels import PriceZone, causal_level_frame, estimate_levels
from .market_data import MarketDataBundle, fetch_market_bundle, synthetic_market_bundle
from .model import ModelDiagnostics, train_classifier
from .news import NewsImpact, YahooNewsProvider, score_news
from .range_model import RangeModel, RangeSummary, train_range_model
from .signals import Guidance, SignalRecommendation, generate_signal
from .spread_model import SpreadModel, SpreadSummary, train_spread_model
from .strategies import (
    STRATEGY_NAMES,
    build_strategy_exposure,
    strategy_parameter_grid,
)


@dataclass(frozen=True, slots=True)
class StrategyLabResult:
    leaderboard: pd.DataFrame
    backtests: dict[str, BacktestResult]
    exposures: pd.DataFrame
    winner: str
    explanation: str
    development_end: pd.Timestamp
    holdout_start: pd.Timestamp


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    bars: pd.DataFrame
    as_of: pd.Timestamp
    current_price: float
    support: PriceZone
    resistance: PriceZone
    probabilities: dict[str, float]
    signal: FusedRecommendation
    base_signal: SignalRecommendation
    emotion: EmotionSnapshot
    news: NewsImpact
    validation: ModelDiagnostics
    range_model: RangeModel
    range_summary: RangeSummary
    spread_model: SpreadModel
    spread_summary: SpreadSummary
    backtest: BacktestResult
    strategy_lab: StrategyLabResult
    sample_count: int
    data_is_stale: bool


def run_analysis(
    lookback_years: int = 15,
    transaction_cost_bps: float = 10.0,
    confidence_margin: float = 0.05,
    *,
    bars: pd.DataFrame | None = None,
    bundle: MarketDataBundle | None = None,
    news_items: list[dict[str, Any]] | None = None,
    include_emotion: bool = True,
    include_news: bool = True,
) -> AnalysisResult:
    """Run causal features, temporal models, current guidance, and OOS backtest."""
    if lookback_years < 1:
        raise ValueError("lookback_years must be positive")
    config = AlphaConfig(transaction_cost_bps=transaction_cost_bps)
    if bundle is not None:
        market = validate_ohlcv(bundle.brkb)
        market_bundle = bundle
    elif bars is None:
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
        start = end - pd.DateOffset(years=lookback_years)
        market_bundle = fetch_market_bundle(config, start=start.date(), end=end.date())
        market = market_bundle.brkb
    else:
        market = validate_ohlcv(bars)
        market_bundle = synthetic_market_bundle(market)
    cutoff = market.index.max() - pd.DateOffset(years=lookback_years)
    market = market.loc[market.index >= cutoff]
    market_bundle = MarketDataBundle(
        market,
        market_bundle.spy.reindex(market.index).ffill(),
        market_bundle.vix.reindex(market.index).ffill(),
    )
    if len(market) < 80:
        raise ValueError("at least 80 daily bars are required")

    levels = causal_level_frame(
        market,
        lookback=min(config.level_lookback, max(20, len(market) // 2)),
        grid_size=128,
        update_every=20,
    )
    features = make_features(market)
    features = features.join(
        pd.DataFrame(
            {
                "support_distance": levels["support"] / market["Close"] - 1,
                "resistance_distance": levels["resistance"] / market["Close"] - 1,
                "level_width_atr": (
                    levels["resistance"] - levels["support"]
                ) / levels["atr"].replace(0, np.nan),
            },
            index=market.index,
        )
    )
    labels = first_touch_labels(market, levels, horizon=config.horizon)
    ranges = future_extrema(market, config.horizon)
    spread_labels = next_day_spread(market)
    classifier = train_classifier(features, labels, config)
    current_price = float(market["Close"].iloc[-1])
    range_model = train_range_model(
        features,
        ranges,
        current_price=current_price,
        config=config,
    )
    spread_model = train_spread_model(
        features,
        spread_labels,
        current_price=current_price,
        config=config,
    )
    support, resistance = estimate_levels(
        market,
        lookback=min(config.level_lookback, max(20, len(market) // 2)),
        atr_period=config.atr_window,
    )
    probabilities = {
        label: float(classifier.probabilities[label])
        for label in ("upper_first", "lower_first", "neither")
    }
    base_signal = generate_signal(
        probabilities,
        current_price,
        support.price,
        resistance.price,
        transaction_cost_bps,
        confidence_margin,
    )
    emotion = compute_emotion(market_bundle, config)
    if news_items is None and bars is None and bundle is None:
        news_items = YahooNewsProvider(config).fetch()
    news = score_news(news_items or [], config)
    signal = fuse_guidance(
        base_signal,
        emotion,
        news,
        confidence_margin,
        config,
        include_emotion=include_emotion,
        include_news=include_news,
    )
    recommendations = _oos_recommendations(
        classifier.oos_probabilities,
        market,
        levels,
        transaction_cost_bps,
        confidence_margin,
    )
    backtest = run_backtest(
        market, recommendations, transaction_cost_bps=transaction_cost_bps
    )
    minimum_risk = (
        ranges["future_min_return"]
        .shift(config.horizon)
        .expanding(min_periods=20)
        .quantile(0.10)
    )
    strategy_lab = _run_strategy_lab(
        market,
        classifier.oos_probabilities,
        levels,
        features,
        minimum_risk,
        transaction_cost_bps,
    )
    as_of = pd.Timestamp(market.index[-1])
    now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
    return AnalysisResult(
        market,
        as_of,
        current_price,
        support,
        resistance,
        probabilities,
        signal,
        base_signal,
        emotion,
        news,
        classifier.validation,
        range_model,
        range_model.summary,
        spread_model,
        spread_model.summary,
        backtest,
        strategy_lab,
        int(labels.notna().sum()),
        bool(now.normalize() - as_of.normalize() > pd.Timedelta(days=5)),
    )


def _oos_recommendations(
    probabilities: pd.DataFrame,
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    transaction_cost_bps: float,
    confidence_margin: float,
) -> pd.Series:
    """Translate only fold-held-out probabilities into dated recommendations."""
    actions: list[str] = []
    for date, row in probabilities.iterrows():
        close = float(bars.at[date, "Close"])
        support = float(levels.at[date, "support"])
        resistance = float(levels.at[date, "resistance"])
        try:
            recommendation = generate_signal(
                row.to_dict(),
                close,
                support,
                resistance,
                transaction_cost_bps,
                confidence_margin,
            )
            actions.append(recommendation.action.value)
        except ValueError:
            actions.append(Guidance.HOLD.value)
    return pd.Series(actions, index=probabilities.index, name="action")


def _run_strategy_lab(
    bars: pd.DataFrame,
    probabilities: pd.DataFrame,
    levels: pd.DataFrame,
    features: pd.DataFrame,
    minimum_risk: pd.Series,
    transaction_cost_bps: float,
) -> StrategyLabResult:
    """Tune on early OOS rows and report only a later untouched holdout."""
    if len(probabilities) < 24:
        raise ValueError("strategy lab requires at least 24 out-of-sample predictions")
    split = min(max(12, int(len(probabilities) * 0.60)), len(probabilities) - 10)
    development_index = probabilities.index[:split]
    holdout_index = probabilities.index[split:]
    development_end = pd.Timestamp(development_index[-1])
    holdout_start = pd.Timestamp(holdout_index[0])
    grids = strategy_parameter_grid()

    selected_parameters: dict[str, dict[str, float]] = {}
    for name in STRATEGY_NAMES:
        scored: list[tuple[bool, float, float, dict[str, float]]] = []
        for parameters in grids[name]:
            exposure = build_strategy_exposure(
                name, bars, probabilities, levels, features, minimum_risk, parameters
            ).loc[development_index]
            result = run_exposure_backtest(
                bars.loc[:development_end],
                exposure,
                transaction_cost_bps=transaction_cost_bps,
            )
            metrics = result.metrics
            drawdown_improvement = (
                metrics["max_drawdown"] - metrics["benchmark_max_drawdown"]
            )
            eligible = (
                metrics["excess_return"] > 0
                and drawdown_improvement >= -1e-12
            )
            scored.append(
                (
                    eligible,
                    metrics["excess_return"],
                    drawdown_improvement,
                    parameters,
                )
            )
        selected_parameters[name] = max(
            scored, key=lambda item: (item[0], item[1], item[2])
        )[3]

    backtests: dict[str, BacktestResult] = {}
    exposure_columns: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for name in STRATEGY_NAMES:
        exposure = build_strategy_exposure(
            name,
            bars,
            probabilities,
            levels,
            features,
            minimum_risk,
            selected_parameters[name],
        ).loc[holdout_index]
        result = run_exposure_backtest(
            bars.loc[holdout_start:],
            exposure,
            transaction_cost_bps=transaction_cost_bps,
        )
        backtests[name] = result
        exposure_columns[name] = exposure
        metrics = result.metrics
        eligible = (
            metrics["excess_return"] > 0
            and metrics["max_drawdown"] >= metrics["benchmark_max_drawdown"] - 1e-12
        )
        rows.append(
            {
                "strategy": name,
                "total_return": metrics["total_return"],
                "buy_hold_return": metrics["buy_hold_return"],
                "excess_return": metrics["excess_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "benchmark_max_drawdown": metrics["benchmark_max_drawdown"],
                "market_exposure": metrics["market_exposure"],
                "turnover": metrics["turnover"],
                "eligible": eligible,
                "parameters": str(selected_parameters[name]),
            }
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["eligible", "excess_return", "max_drawdown"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    eligible_rows = leaderboard[leaderboard["eligible"]]
    if eligible_rows.empty:
        winner = "buy_and_hold"
        explanation = (
            "No candidate beat buy-and-hold after costs without a worse maximum "
            "drawdown on the untouched holdout; buy-and-hold remains the winner."
        )
    else:
        winner = str(eligible_rows.iloc[0]["strategy"])
        explanation = (
            f"{winner} passed the balanced holdout rule: positive excess return "
            "after costs with maximum drawdown no worse than buy-and-hold."
        )
    return StrategyLabResult(
        leaderboard,
        backtests,
        pd.DataFrame(exposure_columns).loc[holdout_index],
        winner,
        explanation,
        development_end,
        holdout_start,
    )


__all__ = ["AnalysisResult", "StrategyLabResult", "run_analysis"]
