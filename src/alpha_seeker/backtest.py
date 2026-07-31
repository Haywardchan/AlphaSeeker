"""Leakage-safe fractional long/cash backtesting with next-open execution."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from .signals import Guidance


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity: pd.DataFrame
    metrics: dict[str, float]
    trades: pd.DataFrame


def run_exposure_backtest(
    bars: pd.DataFrame,
    target_exposure: pd.Series,
    *,
    transaction_cost_bps: float = 0.0,
    initial_capital: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> BacktestResult:
    """Rebalance to close-known 0%-100% targets at the next session open.

    Strategy and benchmark both begin on the first open at which the first
    target is actionable. This makes their starting date and capital identical.
    """
    prices = _validated_bars(bars)
    targets = _validated_exposures(target_exposure)
    _validate_parameters(transaction_cost_bps, initial_capital, risk_free_rate, annualization)
    if targets.empty:
        raise ValueError("target_exposure must contain at least one observation")

    first_position = int(
        np.searchsorted(prices.index.to_numpy(), targets.index[0].to_datetime64(), side="right")
    )
    if first_position >= len(prices):
        raise ValueError("no market session exists after the first exposure signal")
    full_trend = prices["Close"] >= prices["Close"].rolling(200, min_periods=20).mean()
    prices = prices.iloc[first_position:].copy()
    trend = full_trend.reindex(prices.index).fillna(False)
    eligible = _values_known_before_open(prices.index, targets)

    cost_rate = transaction_cost_bps / 10_000.0
    cash, shares = float(initial_capital), 0.0
    benchmark_shares = initial_capital / float(prices["Open"].iloc[0])
    turnover = 0.0
    trade_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for date, row in prices.iterrows():
        open_price, close_price = float(row["Open"]), float(row["Close"])
        pre_trade_equity = cash + shares * open_price
        requested = float(eligible.loc[date])
        current_notional = shares * open_price
        desired_notional = requested * pre_trade_equity
        delta = desired_notional - current_notional
        if delta > 0:
            delta = min(delta, cash / (1.0 + cost_rate))
        cost = abs(delta) * cost_rate
        if delta > 0:
            cash -= delta + cost
        elif delta < 0:
            cash += -delta - cost
        shares += delta / open_price
        if abs(cash) < initial_capital * 1e-12:
            cash = 0.0

        if abs(delta) > pre_trade_equity * 1e-10:
            turnover += abs(delta) / pre_trade_equity
            trade_rows.append(
                {
                    "Date": date,
                    "TargetExposure": requested,
                    "TradedNotional": abs(delta),
                    "TransactionCost": cost,
                }
            )
        close_equity = cash + shares * close_price
        history.append(
            {
                "Date": date,
                "Strategy": close_equity,
                "BuyHold": benchmark_shares * close_price,
                "Position": shares * close_price / close_equity if close_equity > 0 else 0.0,
                "TargetExposure": requested,
                "TradedNotional": abs(delta),
                "TransactionCost": cost,
            }
        )

    equity = pd.DataFrame(history).set_index("Date")
    trades = pd.DataFrame(
        trade_rows,
        columns=["Date", "TargetExposure", "TradedNotional", "TransactionCost"],
    )
    metrics = calculate_metrics(
        equity,
        trades,
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        annualization=annualization,
        turnover=turnover,
        order_count=len(trades),
    )
    strategy_returns = equity["Strategy"].pct_change().fillna(
        equity["Strategy"].iloc[0] / initial_capital - 1.0
    )
    metrics["uptrend_daily_return"] = float(strategy_returns[trend].mean())
    metrics["downtrend_daily_return"] = float(strategy_returns[~trend].mean())
    metrics["market_exposure"] = float(equity["Position"].mean())
    metrics["start_date_ordinal"] = float(equity.index[0].toordinal())
    return BacktestResult(equity=equity, metrics=metrics, trades=trades)


def run_backtest(
    bars: pd.DataFrame,
    recommendations: pd.Series | pd.DataFrame,
    **kwargs: float | int,
) -> BacktestResult:
    """Compatibility wrapper translating BUY/HOLD/SELL into binary exposure."""
    actions = _validated_recommendations(recommendations)
    state = 0.0
    exposures: list[float] = []
    for action in actions:
        if action == Guidance.BUY:
            state = 1.0
        elif action == Guidance.SELL_REDUCE:
            state = 0.0
        exposures.append(state)
    target = pd.Series(exposures, index=actions.index, name="target_exposure")
    return run_exposure_backtest(bars, target, **kwargs)


def backtest(
    bars: pd.DataFrame,
    recommendations: pd.Series | pd.DataFrame,
    **kwargs: float | int,
) -> BacktestResult:
    return run_backtest(bars, recommendations, **kwargs)


def calculate_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    initial_capital: float = 1.0,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
    turnover: float = 0.0,
    order_count: int = 0,
) -> dict[str, float]:
    if equity.empty or not {"Strategy", "BuyHold"}.issubset(equity.columns):
        raise ValueError("equity must contain non-empty Strategy and BuyHold columns")
    strategy, benchmark = equity["Strategy"], equity["BuyHold"]
    total_return = float(strategy.iloc[-1] / initial_capital - 1.0)
    buy_hold_return = float(benchmark.iloc[-1] / initial_capital - 1.0)
    excess = total_return - buy_hold_return
    elapsed_days = (strategy.index[-1] - strategy.index[0]).total_seconds() / 86_400.0
    cagr = (
        float((strategy.iloc[-1] / initial_capital) ** (365.25 / elapsed_days) - 1.0)
        if elapsed_days > 0 and strategy.iloc[-1] > 0
        else total_return
    )
    returns = strategy.pct_change()
    returns.iloc[0] = strategy.iloc[0] / initial_capital - 1.0
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_returns = benchmark.pct_change()
    benchmark_returns.iloc[0] = benchmark.iloc[0] / initial_capital - 1.0
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization) - 1.0
    volatility = float(returns.std(ddof=1))
    benchmark_volatility = float(benchmark_returns.std(ddof=1))
    sharpe = (
        float(sqrt(annualization) * (returns.mean() - daily_rf) / volatility)
        if len(returns) > 1 and volatility > 0
        else 0.0
    )
    benchmark_sharpe = (
        float(
            sqrt(annualization)
            * (benchmark_returns.mean() - daily_rf)
            / benchmark_volatility
        )
        if len(benchmark_returns) > 1 and benchmark_volatility > 0
        else 0.0
    )
    max_drawdown = float((strategy / strategy.cummax().clip(lower=initial_capital) - 1).min())
    benchmark_drawdown = float(
        (benchmark / benchmark.cummax().clip(lower=initial_capital) - 1).min()
    )
    win_rate = (
        float(trades["Won"].mean()) if not trades.empty and "Won" in trades else float("nan")
    )
    return {
        "total_return": total_return,
        "cagr": cagr,
        "excess_total_return": excess,
        "excess_return": excess,
        "sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_drawdown,
        "turnover": float(turnover),
        "trade_count": float(order_count),
        "trades": float(order_count),
        "win_rate": win_rate,
        "buy_hold_return": buy_hold_return,
    }


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("bars must be a non-empty DataFrame")
    lookup = {str(column).lower(): column for column in bars.columns}
    missing = {"open", "high", "low", "close"} - lookup.keys()
    if missing:
        raise ValueError(f"bars is missing OHLC columns: {', '.join(sorted(missing))}")
    result = bars.rename(
        columns={lookup[name]: name.title() for name in ("open", "high", "low", "close")}
    )[["Open", "High", "Low", "Close"]].copy()
    result.index = _datetime_index(result.index, "bars")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("bars index must be unique and increasing")
    result = result.astype(float)
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive")
    if (result["High"] < result[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High must be at least Open, Close, and Low")
    if (result["Low"] > result[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low must be at most Open, Close, and High")
    return result


def _validated_exposures(exposures: pd.Series) -> pd.Series:
    if not isinstance(exposures, pd.Series):
        raise TypeError("target_exposure must be a pandas Series")
    result = pd.to_numeric(exposures.copy(), errors="coerce")
    result.index = _datetime_index(result.index, "target_exposure")
    result = result.dropna().sort_index()
    if result.index.has_duplicates or ((result < 0) | (result > 1)).any():
        raise ValueError("target exposures must be unique, ordered, and in [0, 1]")
    return result.astype(float)


def _validated_recommendations(
    recommendations: pd.Series | pd.DataFrame,
) -> pd.Series:
    if isinstance(recommendations, pd.DataFrame):
        lookup = {str(column).lower(): column for column in recommendations.columns}
        selected = next(
            (
                lookup[name]
                for name in ("action", "recommendation", "guidance", "signal")
                if name in lookup
            ),
            recommendations.columns[0] if len(recommendations.columns) == 1 else None,
        )
        if selected is None:
            raise ValueError("recommendations needs an action column")
        result = recommendations[selected].copy()
    elif isinstance(recommendations, pd.Series):
        result = recommendations.copy()
    else:
        raise TypeError("recommendations must be a pandas Series or DataFrame")
    result.index = _datetime_index(result.index, "recommendations")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("recommendations index must be unique and increasing")
    return result.map(_coerce_action)


def _coerce_action(value: Any) -> Guidance:
    if hasattr(value, "action"):
        value = value.action
    if isinstance(value, Guidance):
        return value
    normalized = str(value).strip().upper().replace(" ", "_").replace("/", "_")
    aliases = {
        "BUY": Guidance.BUY,
        "HOLD": Guidance.HOLD,
        "SELL": Guidance.SELL_REDUCE,
        "REDUCE": Guidance.SELL_REDUCE,
        "SELL_REDUCE": Guidance.SELL_REDUCE,
    }
    if normalized not in aliases:
        raise ValueError(f"unknown recommendation action: {value!r}")
    return aliases[normalized]


def _values_known_before_open(
    dates: pd.DatetimeIndex, values: pd.Series
) -> pd.Series:
    locations = np.searchsorted(values.index.to_numpy(), dates.to_numpy(), side="left") - 1
    return pd.Series(
        [values.iloc[position] if position >= 0 else np.nan for position in locations],
        index=dates,
        dtype=float,
    ).ffill()


def _datetime_index(index: pd.Index, label: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(pd.to_datetime(index))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have a date-like index") from exc
    if result.hasnans:
        raise ValueError(f"{label} index cannot contain missing dates")
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    result.name = index.name
    return result


def _validate_parameters(
    transaction_cost_bps: float,
    initial_capital: float,
    risk_free_rate: float,
    annualization: int,
) -> None:
    if not all(
        np.isfinite(value)
        for value in (transaction_cost_bps, initial_capital, risk_free_rate, annualization)
    ):
        raise ValueError("backtest parameters must be finite")
    if transaction_cost_bps < 0 or initial_capital <= 0 or annualization <= 0:
        raise ValueError("invalid cost, capital, or annualization")
    if risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be greater than -1")


__all__ = [
    "BacktestResult",
    "backtest",
    "calculate_metrics",
    "run_backtest",
    "run_exposure_backtest",
]
