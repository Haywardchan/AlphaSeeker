from datetime import UTC, datetime, timedelta

import numpy as np

from alpha_seeker.config import AlphaConfig
from alpha_seeker.emotion import compute_emotion
from alpha_seeker.market_data import MarketDataBundle, MultiSymbolProvider, synthetic_market_bundle
from alpha_seeker.news import YahooNewsProvider, score_news


def test_synthetic_market_bundle_aligns_symbols(bars):
    bundle = synthetic_market_bundle(bars)
    assert len(bundle.brkb) == len(bars)
    assert bundle.spy.index.equals(bundle.brkb.index)
    assert bundle.vix.index.equals(bundle.brkb.index)


def test_multi_symbol_provider_uses_cache(tmp_path, bars):
    config = AlphaConfig(cache_dir=tmp_path, cache_ttl_seconds=3600)

    def downloader(**kwargs):
        ticker = kwargs["tickers"]
        if ticker == "^VIX":
            frame = bars.copy()
            frame["Close"] = 20.0
            frame["Open"] = 20.0
            frame["High"] = 21.0
            frame["Low"] = 19.0
            return frame
        return bars

    provider = MultiSymbolProvider(config, downloader=downloader)
    first = provider.fetch(force_refresh=True)
    second = provider.fetch(force_refresh=False)
    assert len(first.brkb) == len(second.brkb)
    assert float(second.vix["Close"].iloc[-1]) == 20.0


def test_emotion_composite_is_bounded(bars):
    bundle = synthetic_market_bundle(bars)
    snapshot = compute_emotion(bundle)
    assert -1.0 <= snapshot.composite_score <= 1.0
    assert snapshot.trin_proxy > 0
    assert not snapshot.history.empty


def test_emotion_fear_on_vix_spike(bars):
    bundle = synthetic_market_bundle(bars)
    vix = bundle.vix.copy()
    vix.iloc[-30:, vix.columns.get_loc("Close")] = np.linspace(15, 45, 30)
    fear_bundle = MarketDataBundle(bundle.brkb, bundle.spy, vix)
    snapshot = compute_emotion(fear_bundle)
    assert snapshot.composite_score < 0.2


def _headline(
    title: str,
    hours_ago: float = 1.0,
    summary: str = "Berkshire Hathaway update.",
) -> dict:
    published = datetime.now(UTC) - timedelta(hours=hours_ago)
    return {
        "content": {
            "title": title,
            "summary": summary,
            "pubDate": published.isoformat().replace("+00:00", "Z"),
            "provider": "TestWire",
            "canonicalUrl": "https://example.com/news",
        }
    }


def test_score_news_positive_headline():
    impact = score_news([_headline("Berkshire Hathaway announces buyback and earnings beat")])
    assert impact.aggregate_sentiment > 0
    assert impact.impact_score > 0
    assert impact.confidence > 0


def test_score_news_filters_irrelevant_headline():
    impact = score_news([
        _headline(
            "Generic market wrap with no company mention",
            hours_ago=1,
            summary="Macro overview only.",
        )
    ])
    assert impact.articles == ()
    assert impact.impact_score == 0.0


def test_score_news_empty_provider():
    impact = score_news([])
    assert impact.summary.startswith("No recent relevant headlines")


def test_yahoo_news_provider_cache(tmp_path):
    config = AlphaConfig(cache_dir=tmp_path, news_cache_ttl_seconds=3600)
    calls = {"count": 0}

    def fetcher(_symbol: str, _count: int) -> list[dict]:
        calls["count"] += 1
        return [_headline("Buffett comments on Berkshire outlook")]

    provider = YahooNewsProvider(config, fetcher=fetcher)
    first = provider.fetch(force_refresh=True)
    second = provider.fetch(force_refresh=False)
    assert len(first) == 1
    assert len(second) == 1
    assert calls["count"] == 1
