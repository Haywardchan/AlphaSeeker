"""Multi-symbol market data bundle for a primary instrument, SPY, and VIX."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, AlphaConfig
from .data import OHLCV_COLUMNS, validate_ohlcv

Downloader = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class MarketDataBundle:
    """Aligned daily OHLCV frames for primary and auxiliary symbols."""

    primary: pd.DataFrame
    spy: pd.DataFrame
    vix: pd.DataFrame

    @property
    def brkb(self) -> pd.DataFrame:
        """Compatibility alias for callers using the original BRK-specific name."""
        return self.primary


class MultiSymbolProvider:
    """Fetch and cache OHLCV for multiple Yahoo Finance symbols."""

    def __init__(
        self,
        config: AlphaConfig = DEFAULT_CONFIG,
        *,
        downloader: Downloader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._downloader = downloader
        self._clock = clock

    def fetch(
        self,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        *,
        force_refresh: bool = False,
    ) -> MarketDataBundle:
        symbols = (self.config.ticker, self.config.market_proxy, self.config.fear_gauge)
        frames = {
            symbol: self._fetch_symbol(symbol, start, end, force_refresh=force_refresh)
            for symbol in symbols
        }
        return MarketDataBundle(
            frames[self.config.ticker],
            frames[self.config.market_proxy],
            frames[self.config.fear_gauge],
        )

    def _fetch_symbol(
        self,
        symbol: str,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
        *,
        force_refresh: bool,
    ) -> pd.DataFrame:
        path = self._cache_path(symbol, start, end)
        if not force_refresh and (cached := self._read_cache(path)) is not None:
            return cached.copy()
        data = validate_ohlcv(self._download(symbol, start, end))
        self._write_cache(path, data)
        return data.copy()

    def _cache_path(
        self,
        symbol: str,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> Path:
        ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)
        return self.config.cache_dir / (
            f"{ticker}_{self.config.interval}_{_date_key(start, 'begin')}_"
            f"{_date_key(end, 'latest')}.pkl"
        )

    def _download(
        self,
        symbol: str,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> pd.DataFrame:
        downloader = self._downloader
        if downloader is None:
            try:
                import yfinance as yf
            except ImportError as exc:
                raise RuntimeError("yfinance is required to download data") from exc
            downloader = yf.download
        result = downloader(
            tickers=symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError("downloader must return a DataFrame")
        if result.empty:
            raise ValueError(f"no data returned for {symbol}")
        return _normalize_symbol_frame(result, symbol)

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.is_file():
            return None
        ttl = self.config.cache_ttl_seconds
        if ttl is not None and self._clock() - path.stat().st_mtime > ttl:
            return None
        try:
            return validate_ohlcv(pd.read_pickle(path))
        except (OSError, ValueError, TypeError, EOFError, ImportError):
            return None

    @staticmethod
    def _write_cache(path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pkl.tmp")
        data.to_pickle(temporary)
        temporary.replace(path)


def fetch_market_bundle(
    config: AlphaConfig | None = None,
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    downloader: Downloader | None = None,
    force_refresh: bool = False,
) -> MarketDataBundle:
    """Download the primary instrument, SPY, and VIX with shared caching."""
    return MultiSymbolProvider(config or DEFAULT_CONFIG, downloader=downloader).fetch(
        start, end, force_refresh=force_refresh
    )


def synthetic_market_bundle(primary: pd.DataFrame) -> MarketDataBundle:
    """Build auxiliary frames from injected primary bars for offline analysis."""
    market = validate_ohlcv(primary)
    spy = market.copy()
    spy["Close"] = market["Close"] * 0.45
    spy["Open"] = market["Open"] * 0.45
    spy["High"] = market["High"] * 0.45
    spy["Low"] = market["Low"] * 0.45
    spy["Volume"] = market["Volume"] * 8
    vix = market.copy()
    vix_vol = market["Close"].pct_change(fill_method=None).rolling(20).std() * 400
    vix["Close"] = (14 + vix_vol).fillna(14)
    vix["Open"] = vix["Close"]
    vix["High"] = vix["Close"] * 1.05
    vix["Low"] = vix["Close"] * 0.95
    vix["Volume"] = 0
    return MarketDataBundle(market, validate_ohlcv(spy), validate_ohlcv(vix))


def _normalize_symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        fields = {str(name).lower() for name in OHLCV_COLUMNS}
        for field_level in range(data.columns.nlevels):
            values = {
                str(value).lower()
                for value in data.columns.get_level_values(field_level)
            }
            if not fields.issubset(values):
                continue
            selected = data
            for level in range(data.columns.nlevels):
                if level == field_level:
                    continue
                choices = data.columns.get_level_values(level).unique()
                match = next(
                    (value for value in choices if str(value) in {symbol, symbol.upper()}),
                    None,
                )
                if match is not None:
                    selected = selected.xs(match, axis=1, level=level)
                    break
            if isinstance(selected.columns, pd.MultiIndex):
                selected.columns = selected.columns.get_level_values(field_level)
            data = selected
            break
    lookup = {str(column).strip().lower(): column for column in data.columns}
    missing = [name for name in OHLCV_COLUMNS if name.lower() not in lookup]
    if missing:
        raise ValueError(f"missing OHLCV columns for {symbol}: {', '.join(missing)}")
    data = data[[lookup[name.lower()] for name in OHLCV_COLUMNS]]
    data.columns = list(OHLCV_COLUMNS)
    return data


def _date_key(value: str | date | datetime | None, default: str) -> str:
    return default if value is None else pd.Timestamp(value).strftime("%Y%m%d")


__all__ = [
    "MarketDataBundle",
    "MultiSymbolProvider",
    "fetch_market_bundle",
    "synthetic_market_bundle",
]
