"""Local NLP news fetching and impact scoring for BRK.B."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp
from pathlib import Path
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import DEFAULT_CONFIG, AlphaConfig

RELEVANCE_TERMS = (
    "brk",
    "brk.b",
    "brk-b",
    "berkshire",
    "buffett",
    "hathaway",
)
POSITIVE_TERMS = (
    "earnings beat",
    "buyback",
    "upgrade",
    "outperform",
    "record profit",
    "raises guidance",
)
NEGATIVE_TERMS = (
    "downgrade",
    "lawsuit",
    "investigation",
    "misses estimates",
    "recall",
    "probe",
    "cuts guidance",
)


@dataclass(frozen=True, slots=True)
class NewsArticle:
    title: str
    publisher: str
    published_at: pd.Timestamp
    sentiment: float
    relevance: float
    link: str


@dataclass(frozen=True, slots=True)
class NewsImpact:
    articles: tuple[NewsArticle, ...]
    aggregate_sentiment: float
    impact_score: float
    confidence: float
    summary: str


class YahooNewsProvider:
    """Fetch recent BRK.B headlines from Yahoo Finance."""

    def __init__(
        self,
        config: AlphaConfig = DEFAULT_CONFIG,
        *,
        fetcher: Callable[[str, int], list[dict[str, Any]]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._fetcher = fetcher
        self._clock = clock
        self._analyzer = SentimentIntensityAnalyzer()

    def fetch(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        path = self.config.cache_dir / "news_brkb.json"
        if not force_refresh and (cached := self._read_cache(path)) is not None:
            return cached
        items = self._download()
        self._write_cache(path, items)
        return items

    def _download(self) -> list[dict[str, Any]]:
        if self._fetcher is not None:
            return self._fetcher(self.config.ticker, self.config.news_max_articles)
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required to download news") from exc
        return yf.Ticker(self.config.ticker).get_news(
            count=self.config.news_max_articles,
            tab="news",
        )

    def _read_cache(self, path: Path) -> list[dict[str, Any]] | None:
        if not path.is_file():
            return None
        if self._clock() - path.stat().st_mtime > self.config.news_cache_ttl_seconds:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, list) else None

    @staticmethod
    def _write_cache(path: Path, items: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items), encoding="utf-8")


def score_news(
    items: Sequence[dict[str, Any]],
    config: AlphaConfig | None = None,
    *,
    now: datetime | None = None,
) -> NewsImpact:
    """Score headlines into a bounded impact estimate."""
    cfg = config or DEFAULT_CONFIG
    analyzer = SentimentIntensityAnalyzer()
    current = now or datetime.now(UTC)
    articles: list[NewsArticle] = []
    for item in items:
        parsed = _parse_article(item, analyzer, current, cfg)
        if parsed is not None:
            articles.append(parsed)
    if not articles:
        return NewsImpact(
            (),
            0.0,
            0.0,
            0.0,
            "No recent relevant headlines were available; news impact is neutral.",
        )
    weights = [_article_weight(article, cfg) for article in articles]
    total_weight = sum(weights)
    aggregate = sum(
        article.sentiment * weight
        for article, weight in zip(articles, weights, strict=True)
    )
    aggregate_sentiment = float(
        np_clip(aggregate / total_weight if total_weight else 0.0, -1.0, 1.0)
    )
    confidence = _news_confidence(articles, aggregate_sentiment)
    impact_score = float(
        np_clip(
            aggregate_sentiment * confidence,
            -cfg.news_impact_cap,
            cfg.news_impact_cap,
        )
    )
    summary = _impact_summary(aggregate_sentiment, impact_score, len(articles))
    return NewsImpact(
        tuple(articles),
        aggregate_sentiment,
        impact_score,
        confidence,
        summary,
    )


def _parse_article(
    item: dict[str, Any],
    analyzer: SentimentIntensityAnalyzer,
    now: datetime,
    config: AlphaConfig,
) -> NewsArticle | None:
    content = item.get("content", item)
    title = str(content.get("title", "")).strip()
    if not title:
        return None
    summary = str(content.get("summary", "")).strip()
    publisher = str(content.get("provider", content.get("publisher", "Unknown"))).strip()
    link = str(
        content.get(
            "canonicalUrl",
            content.get("clickThroughUrl", content.get("link", "")),
        )
    ).strip()
    published = _parse_timestamp(content.get("pubDate", content.get("displayTime")))
    text = f"{title}. {summary}".strip()
    relevance = _relevance_score(text)
    if relevance <= 0.0:
        return None
    sentiment = _headline_sentiment(text, analyzer)
    return NewsArticle(title, publisher, published, sentiment, relevance, link)


def _headline_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    base = analyzer.polarity_scores(text)["compound"]
    lowered = text.lower()
    adjustment = 0.0
    for term in POSITIVE_TERMS:
        if term in lowered:
            adjustment += 0.15
    for term in NEGATIVE_TERMS:
        if term in lowered:
            adjustment -= 0.15
    return float(np_clip(base + adjustment, -1.0, 1.0))


def _relevance_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for term in RELEVANCE_TERMS if term in lowered)
    if hits == 0:
        return 0.0
    return float(min(1.0, 0.5 + 0.25 * hits))


def _article_weight(article: NewsArticle, config: AlphaConfig) -> float:
    published = article.published_at
    if published.tzinfo is None:
        published = published.tz_localize("UTC")
    else:
        published = published.tz_convert("UTC")
    age_hours = max(
        0.0,
        (pd.Timestamp.now(tz="UTC") - published).total_seconds() / 3600,
    )
    decay = exp(-age_hours / config.news_half_life_hours)
    return article.relevance * decay


def _news_confidence(articles: Sequence[NewsArticle], aggregate_sentiment: float) -> float:
    if not articles:
        return 0.0
    count_factor = min(1.0, len(articles) / 5.0)
    dispersion = pd.Series([article.sentiment for article in articles]).std(ddof=0)
    agreement = 1.0 - min(1.0, float(dispersion if pd.notna(dispersion) else 0.0))
    direction = abs(aggregate_sentiment)
    return float(np_clip(0.35 * count_factor + 0.35 * agreement + 0.30 * direction, 0.0, 1.0))


def _impact_summary(aggregate_sentiment: float, impact_score: float, count: int) -> str:
    tone = "neutral"
    if aggregate_sentiment > 0.15:
        tone = "supportive"
    elif aggregate_sentiment < -0.15:
        tone = "cautious"
    return (
        f"{count} relevant headline(s) with {tone} tone; "
        f"news impact adjusts expected edge by {impact_score:+.2%}."
    )


def _parse_timestamp(value: Any) -> pd.Timestamp:
    if value in (None, ""):
        return pd.Timestamp(datetime.now(UTC)).tz_localize(None)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def np_clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = [
    "NewsArticle",
    "NewsImpact",
    "YahooNewsProvider",
    "score_news",
]
