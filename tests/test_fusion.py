import pandas as pd

from alpha_seeker.config import AlphaConfig
from alpha_seeker.emotion import EmotionRegime, EmotionSnapshot
from alpha_seeker.fusion import fuse_guidance
from alpha_seeker.news import NewsImpact
from alpha_seeker.signals import Guidance, SignalRecommendation


def _base(action: Guidance = Guidance.HOLD) -> SignalRecommendation:
    edge = 0.01 if action is Guidance.BUY else -0.01 if action is Guidance.SELL_REDUCE else 0.0
    return SignalRecommendation(
        action=action,
        edge=edge,
        weighted_upside=0.03,
        weighted_downside=0.02,
        gross_edge=0.012,
        estimated_cost=0.001,
        confidence_spread=0.08,
        rationale="base rationale",
    )


def _emotion(score: float = 0.0, vix_zscore: float = 0.0) -> EmotionSnapshot:
    return EmotionSnapshot(
        composite_score=score,
        regime=EmotionRegime.NEUTRAL,
        components={"vix_zscore": vix_zscore},
        trin_proxy=1.0,
        as_of=pd.Timestamp("2024-01-02"),
        history=pd.DataFrame({"composite_score": [score], "vix_close": [20.0]}),
    )


def _news(sentiment: float = 0.0, confidence: float = 0.8) -> NewsImpact:
    return NewsImpact((), sentiment, sentiment * 0.01, confidence, "news summary")


def test_fusion_supports_buy_from_hold_with_positive_adjustments():
    fused = fuse_guidance(_base(), _emotion(0.5), _news(0.4), confidence_margin=0.05)
    assert fused.final_action is Guidance.BUY
    assert fused.emotion_adjustment > 0


def test_fusion_blocks_buy_under_vix_stress():
    config = AlphaConfig(vix_stress_zscore=1.5)
    fused = fuse_guidance(
        _base(Guidance.HOLD),
        _emotion(score=-0.6, vix_zscore=2.5),
        _news(0.2),
        confidence_margin=0.05,
        config=config,
    )
    assert fused.final_action is Guidance.HOLD
    assert fused.gates_applied


def test_fusion_blocks_buy_on_negative_news():
    fused = fuse_guidance(
        SignalRecommendation(
            action=Guidance.HOLD,
            edge=0.0,
            weighted_upside=0.01,
            weighted_downside=0.009,
            gross_edge=0.0015,
            estimated_cost=0.001,
            confidence_spread=0.08,
            rationale="base rationale",
        ),
        _emotion(0.1),
        NewsImpact((), -0.8, -0.012, 0.9, "negative news"),
        confidence_margin=0.05,
    )
    assert fused.final_action is Guidance.HOLD
    assert any("Negative news gate" in gate for gate in fused.gates_applied)


def test_fusion_preserves_base_buy():
    fused = fuse_guidance(_base(Guidance.BUY), _emotion(-0.2), _news(-0.1), 0.05)
    assert fused.final_action is Guidance.BUY
    assert fused.base_action is Guidance.BUY
