"""Fuse technical, emotion, and news inputs into final position advice."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_CONFIG, AlphaConfig
from .emotion import EmotionSnapshot
from .news import NewsImpact
from .signals import Guidance, SignalRecommendation


@dataclass(frozen=True, slots=True)
class FusedRecommendation:
    base_action: Guidance
    final_action: Guidance
    edge: float
    adjusted_edge: float
    base_edge: float
    emotion_adjustment: float
    news_adjustment: float
    confidence_margin_used: float
    rationale: str
    gates_applied: tuple[str, ...]

    @property
    def action(self) -> Guidance:
        return self.final_action


def fuse_guidance(
    base: SignalRecommendation,
    emotion: EmotionSnapshot,
    news: NewsImpact,
    confidence_margin: float,
    config: AlphaConfig | None = None,
    *,
    include_emotion: bool = True,
    include_news: bool = True,
) -> FusedRecommendation:
    """Merge base technical guidance with emotion and news overlays."""
    cfg = config or DEFAULT_CONFIG
    buy_edge = base.gross_edge - base.estimated_cost
    reduce_edge = -base.gross_edge - base.estimated_cost
    emotion_adj = (
        cfg.emotion_weight * emotion.composite_score * cfg.emotion_scale
        if include_emotion
        else 0.0
    )
    news_adj = cfg.news_weight * news.impact_score if include_news else 0.0
    adjusted_buy_edge = buy_edge + emotion_adj + news_adj
    adjusted_reduce_edge = reduce_edge - emotion_adj - news_adj
    margin = confidence_margin
    gates: list[str] = []

    vix_zscore = float(emotion.components.get("vix_zscore", 0.0))
    if (
        include_emotion
        and vix_zscore > cfg.vix_stress_zscore
        and emotion.composite_score < -0.4
    ):
        margin = confidence_margin * 2.0
        gates.append("VIX stress gate widened confidence margin for new BUY")

    if (
        include_news
        and news.aggregate_sentiment < -0.5
        and news.confidence >= 0.5
        and adjusted_buy_edge <= 2.0 * base.estimated_cost
    ):
        gates.append("Negative news gate blocked new BUY")

    final_action, adjusted_edge = _resolve_action(
        base,
        adjusted_buy_edge,
        adjusted_reduce_edge,
        margin,
        cfg,
        gates,
    )
    rationale = (
        f"{final_action.value}: base guidance was {base.action.value} "
        f"({base.edge:+.2%} edge). Emotion adjustment {emotion_adj:+.2%}, "
        f"news adjustment {news_adj:+.2%}; adjusted buy edge "
        f"{adjusted_buy_edge:+.2%}, adjusted reduce edge "
        f"{adjusted_reduce_edge:+.2%}. Confidence margin used "
        f"{margin:.1%}. {emotion.regime.value} mood ({emotion.composite_score:+.2f}); "
        f"{news.summary}"
    )
    if gates:
        rationale += " Gates: " + "; ".join(gates) + "."
    return FusedRecommendation(
        base.action,
        final_action,
        adjusted_edge,
        adjusted_edge,
        base.edge,
        emotion_adj,
        news_adj,
        margin,
        rationale,
        tuple(gates),
    )


def _resolve_action(
    base: SignalRecommendation,
    adjusted_buy_edge: float,
    adjusted_reduce_edge: float,
    margin: float,
    config: AlphaConfig,
    gates: list[str],
) -> tuple[Guidance, float]:
    spread = base.confidence_spread
    flip_threshold = config.fusion_flip_cost_multiple * base.estimated_cost
    support_threshold = margin
    if not gates:
        support_threshold = margin * config.fusion_support_factor

    if base.action is Guidance.BUY:
        if "Negative news gate blocked new BUY" in gates:
            return Guidance.HOLD, adjusted_buy_edge
        return Guidance.BUY, adjusted_buy_edge

    if base.action is Guidance.SELL_REDUCE:
        return Guidance.SELL_REDUCE, -adjusted_reduce_edge

    if (
        spread >= support_threshold
        and adjusted_buy_edge > 0.0
        and "Negative news gate blocked new BUY" not in gates
    ):
        return Guidance.BUY, adjusted_buy_edge

    if -spread >= support_threshold and adjusted_reduce_edge > 0.0:
        return Guidance.SELL_REDUCE, -adjusted_reduce_edge

    if (
        base.action is Guidance.BUY
        and adjusted_reduce_edge > flip_threshold
        and -spread >= margin
    ):
        return Guidance.SELL_REDUCE, -adjusted_reduce_edge

    if (
        base.action is Guidance.SELL_REDUCE
        and adjusted_buy_edge > flip_threshold
        and spread >= margin
        and "Negative news gate blocked new BUY" not in gates
    ):
        return Guidance.BUY, adjusted_buy_edge

    edge = adjusted_buy_edge if adjusted_buy_edge >= adjusted_reduce_edge else -adjusted_reduce_edge
    return Guidance.HOLD, edge


__all__ = ["FusedRecommendation", "fuse_guidance"]
