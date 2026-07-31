"""Transparent probability- and level-based research guidance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
import pandas as pd


class Guidance(StrEnum):
    """Actions emitted by the advisory rule."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL_REDUCE = "SELL_REDUCE"


BUY = Guidance.BUY.value
HOLD = Guidance.HOLD.value
SELL_REDUCE = Guidance.SELL_REDUCE.value
SELL = SELL_REDUCE  # Backwards-compatible constant; emitted text remains explicit.


@dataclass(frozen=True, slots=True)
class SignalRecommendation:
    """A recommendation together with the quantities that produced it."""

    action: Guidance
    edge: float
    weighted_upside: float
    weighted_downside: float
    gross_edge: float
    estimated_cost: float
    confidence_spread: float
    rationale: str

    @property
    def guidance(self) -> Guidance:
        """Alias for callers that name the output field ``guidance``."""
        return self.action


AdvisorySignal = SignalRecommendation


def generate_signal(
    probabilities: Mapping[str, float],
    current_price: float,
    support_price: float,
    resistance_price: float,
    transaction_cost_bps: float = 0.0,
    confidence_margin: float = 0.05,
) -> SignalRecommendation:
    """Return balanced BUY, HOLD, or SELL_REDUCE guidance.

    Upside and downside are first-touch probabilities multiplied by returns to
    resistance and support. Directional guidance requires a positive edge
    after estimated round-trip cost and a probability lead at least as large
    as ``confidence_margin``. Identical tests are applied in both directions.
    """

    upper = _probability(probabilities, "upper_first", "p_upper", "upper")
    lower = _probability(probabilities, "lower_first", "p_lower", "lower")
    neither = _probability(probabilities, "neither", "p_neither")
    _validate_inputs(
        upper,
        lower,
        neither,
        current_price,
        support_price,
        resistance_price,
        transaction_cost_bps,
        confidence_margin,
    )

    upside_return = (resistance_price - current_price) / current_price
    downside_return = (current_price - support_price) / current_price
    weighted_upside = upper * upside_return
    weighted_downside = lower * downside_return
    gross_edge = weighted_upside - weighted_downside
    estimated_cost = transaction_cost_bps / 10_000.0
    confidence_spread = upper - lower
    buy_edge = gross_edge - estimated_cost
    reduce_edge = -gross_edge - estimated_cost

    if confidence_spread >= confidence_margin and buy_edge > 0.0:
        action = Guidance.BUY
        edge = buy_edge
        decision = "upside leads after cost and meets the confidence margin"
    elif -confidence_spread >= confidence_margin and reduce_edge > 0.0:
        action = Guidance.SELL_REDUCE
        edge = -reduce_edge
        decision = "downside leads after cost and meets the confidence margin"
    else:
        action = Guidance.HOLD
        edge = gross_edge - estimated_cost if gross_edge >= 0.0 else gross_edge + estimated_cost
        decision = "neither direction clears both cost and the confidence margin"

    rationale = (
        f"{action.value}: probability-weighted upside {weighted_upside:.2%} "
        f"(P(upper) {upper:.1%} × level upside {upside_return:.2%}) versus "
        f"downside {weighted_downside:.2%} "
        f"(P(lower) {lower:.1%} × level downside {downside_return:.2%}); "
        f"estimated round-trip cost {estimated_cost:.2%}, upper-minus-lower "
        f"confidence spread {confidence_spread:+.1%} versus required "
        f"{confidence_margin:.1%}. P(neither) is {neither:.1%}; {decision}."
    )
    return SignalRecommendation(
        action=action,
        edge=edge,
        weighted_upside=weighted_upside,
        weighted_downside=weighted_downside,
        gross_edge=gross_edge,
        estimated_cost=estimated_cost,
        confidence_spread=confidence_spread,
        rationale=rationale,
    )


def advise(
    upper_probability: float,
    lower_probability: float,
    neither_probability: float,
    current_price: float,
    support_price: float,
    resistance_price: float,
    transaction_cost_bps: float = 0.0,
    confidence_margin: float = 0.05,
) -> SignalRecommendation:
    """Convenience wrapper for separately stored probabilities."""

    return generate_signal(
        {
            "upper_first": upper_probability,
            "lower_first": lower_probability,
            "neither": neither_probability,
        },
        current_price,
        support_price,
        resistance_price,
        transaction_cost_bps,
        confidence_margin,
    )


def expected_value_signals(
    probabilities: pd.DataFrame | np.ndarray,
    *,
    upper_return: float = 0.04,
    lower_return: float = -0.04,
    probability_threshold: float = 0.45,
    confidence_margin: float = 0.0,
    min_expected_return: float = 0.0,
    transaction_cost_bps: float = 5.0,
    expected_downside: pd.Series | np.ndarray | None = None,
    expected_upside: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Vectorized compatibility API for dated model probabilities.

    Array columns are lower, neither, upper. Named frames may instead provide
    ``p_lower``, ``p_neither``, and ``p_upper``. Downside values may be negative
    returns; output additionally exposes their positive weighted magnitude.
    """

    if not lower_return < 0.0 < upper_return:
        raise ValueError("lower_return must be negative and upper_return positive")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1]")
    if not 0.0 <= confidence_margin <= 1.0:
        raise ValueError("confidence_margin must be in [0, 1]")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative")

    if isinstance(probabilities, pd.DataFrame):
        columns = ["p_lower", "p_neither", "p_upper"]
        probability = (
            probabilities[columns].to_numpy(dtype=float)
            if set(columns).issubset(probabilities.columns)
            else probabilities.to_numpy(dtype=float)
        )
        index = probabilities.index
    else:
        probability = np.asarray(probabilities, dtype=float)
        index = pd.RangeIndex(len(probability))
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError("probabilities must have three class columns")
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sum = probability.sum(axis=1)
    if (row_sum <= 0.0).any():
        raise ValueError("each probability row must have positive mass")
    probability = probability / row_sum[:, None]
    p_lower, p_neither, p_upper = probability.T

    downside = _range_array(expected_downside, len(probability), lower_return, "downside")
    upside = _range_array(expected_upside, len(probability), upper_return, "upside")
    weighted_upside = p_upper * np.maximum(upside, 0.0)
    weighted_downside = p_lower * np.abs(np.minimum(downside, 0.0))
    gross_edge = weighted_upside - weighted_downside
    cost = transaction_cost_bps / 10_000.0
    buy_edge = gross_edge - cost
    reduce_edge = -gross_edge - cost
    spread = p_upper - p_lower
    buy = (
        (p_upper >= probability_threshold)
        & (spread >= confidence_margin)
        & (buy_edge > min_expected_return)
    )
    reduce = (
        ~buy
        & (p_lower >= probability_threshold)
        & (-spread >= confidence_margin)
        & (reduce_edge > min_expected_return)
    )
    action = np.select([buy, reduce], [BUY, SELL_REDUCE], default=HOLD)
    return pd.DataFrame(
        {
            "signal": action,
            "action": action,
            "expected_return": gross_edge,
            "net_expected_return": gross_edge - cost,
            "weighted_upside": weighted_upside,
            "weighted_downside": weighted_downside,
            "confidence": np.maximum(p_lower, p_upper),
            "confidence_spread": spread,
            "p_lower": p_lower,
            "p_neither": p_neither,
            "p_upper": p_upper,
        },
        index=index,
    )


generate_signals = expected_value_signals


def _range_array(
    values: pd.Series | np.ndarray | None,
    length: int,
    default: float,
    label: str,
) -> np.ndarray:
    result = np.full(length, default) if values is None else np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"expected {label} must be a finite vector matching probability rows")
    return result


def _probability(probabilities: Mapping[str, float], *names: str) -> float:
    for name in names:
        if name in probabilities:
            return float(probabilities[name])
    raise KeyError(f"probabilities must contain one of: {', '.join(names)}")


def _validate_inputs(
    upper: float,
    lower: float,
    neither: float,
    current_price: float,
    support_price: float,
    resistance_price: float,
    transaction_cost_bps: float,
    confidence_margin: float,
) -> None:
    values = (
        upper,
        lower,
        neither,
        current_price,
        support_price,
        resistance_price,
        transaction_cost_bps,
        confidence_margin,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("all advisory inputs must be finite")
    if any(probability < 0.0 or probability > 1.0 for probability in (upper, lower, neither)):
        raise ValueError("probabilities must be in [0, 1]")
    if abs(upper + lower + neither - 1.0) > 1e-6:
        raise ValueError("upper, lower and neither probabilities must sum to 1")
    if not 0.0 < support_price < current_price < resistance_price:
        raise ValueError("prices must satisfy 0 < support < current < resistance")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative")
    if not 0.0 <= confidence_margin <= 1.0:
        raise ValueError("confidence_margin must be in [0, 1]")


__all__ = [
    "AdvisorySignal",
    "BUY",
    "Guidance",
    "HOLD",
    "SELL",
    "SELL_REDUCE",
    "SignalRecommendation",
    "advise",
    "expected_value_signals",
    "generate_signal",
    "generate_signals",
]
