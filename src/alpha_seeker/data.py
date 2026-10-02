"""Validated yfinance acquisition and deterministic OHLCV caching."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, SUPPORTED_SYMBOLS, AlphaConfig

BRK_SYMBOL = "BRK-B"
OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
OHLC_RELATIVE_TOLERANCE = 1e-10
Downloader = Callable[..., pd.DataFrame]


def normalize_symbol(symbol: str) -> str:
    """Normalize supported Yahoo ticker spellings."""
    normalized = symbol.strip().upper().replace(".", "-")
    if normalized not in SUPPORTED_SYMBOLS:
        supported = ", ".join(SUPPORTED_SYMBOLS)
        raise ValueError(f"symbol must be one of {supported}, got {symbol!r}")
    return normalized


def normalize_ohlcv(frame: pd.DataFrame, ticker: str = BRK_SYMBOL) -> pd.DataFrame:
    """Normalize yfinance single- or multi-index output to daily OHLCV."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data = _flatten_columns(data, ticker)
    lookup = {str(column).strip().lower(): column for column in data.columns}
    missing = [name for name in OHLCV_COLUMNS if name.lower() not in lookup]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")
    data = data[[lookup[name.lower()] for name in OHLCV_COLUMNS]]
    data.columns = list(OHLCV_COLUMNS)
    data = data.apply(pd.to_numeric, errors="coerce")
    index = pd.to_datetime(data.index, errors="coerce", utc=True)
    valid = ~index.isna()
    data = data.loc[valid]
    data.index = pd.DatetimeIndex(
        index[valid].tz_convert(None).normalize(), name="Date"
    )
    data = data[~data.index.duplicated(keep="last")].sort_index()
    return data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Open", "High", "Low", "Close"]
    ).astype({name: "float64" for name in OHLCV_COLUMNS})


def validate_ohlcv(frame: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    """Return normalized data or raise when market invariants are violated."""
    data = normalize_ohlcv(frame)
    if data.empty and not allow_empty:
        raise ValueError("OHLCV data is empty")
    prices = data[["Open", "High", "Low", "Close"]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("prices must be finite and positive")
    highest_other = prices[["Open", "Low", "Close"]].max(axis=1)
    invalid_high = (data["High"] < highest_other) & ~np.isclose(
        data["High"],
        highest_other,
        rtol=OHLC_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    if invalid_high.any():
        raise ValueError("High is below another daily price")
    lowest_other = prices[["Open", "High", "Close"]].min(axis=1)
    invalid_low = (data["Low"] > lowest_other) & ~np.isclose(
        data["Low"],
        lowest_other,
        rtol=OHLC_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    if invalid_low.any():
        raise ValueError("Low is above another daily price")
    if data["Volume"].dropna().lt(0).any():
        raise ValueError("Volume must be non-negative")
    return data


class YFinanceProvider:
    """Fetch daily bars with validated, TTL-aware pickle caching."""

    def __init__(
        self,
        config: AlphaConfig = DEFAULT_CONFIG,
        *,
        downloader: Downloader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config, self._downloader, self._clock = config, downloader, clock

    def get(
        self,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        *,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return an independent normalized frame for the requested range."""
        path = self.cache_path(start, end)
        if not force_refresh and (cached := self._read_cache(path)) is not None:
            return cached.copy()
        data = validate_ohlcv(self._download(start, end))
        self._write_cache(path, data)
        return data.copy()

    def cache_path(
        self,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> Path:
        """Return a stable cache location for a date-range request."""
        ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.config.ticker)
        return self.config.cache_dir / (
            f"{ticker}_{self.config.interval}_{_date_key(start, 'begin')}_"
            f"{_date_key(end, 'latest')}.pkl"
        )

    def _download(
        self,
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
            tickers=self.config.ticker, start=start, end=end,
            interval="1d", auto_adjust=True, actions=False, progress=False,
            group_by="column", threads=False,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError("downloader must return a DataFrame")
        return result

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


def download_ohlcv(
    symbol: str = BRK_SYMBOL,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    period: str | None = "max",
    progress: bool = False,
    downloader: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Download adjusted bars for a supported primary instrument without caching."""
    ticker = normalize_symbol(symbol)
    if downloader is None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required to download data") from exc
        downloader = yf.download
    kwargs: dict[str, Any] = dict(
        tickers=ticker, interval="1d", auto_adjust=True, actions=False,
        progress=progress, group_by="column", threads=False,
    )
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if start is None and period is not None:
        kwargs["period"] = period
    return validate_ohlcv(pd.DataFrame(downloader(**kwargs)))


get_data = download_ohlcv


def _flatten_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    fields = {name.lower() for name in OHLCV_COLUMNS}
    for field_level in range(frame.columns.nlevels):
        values = {
            str(value).lower()
            for value in frame.columns.get_level_values(field_level)
        }
        if not fields.issubset(values):
            continue
        selected = frame
        for level in range(frame.columns.nlevels):
            if level == field_level:
                continue
            choices = frame.columns.get_level_values(level).unique()
            match = next((value for value in choices if str(value) == ticker), None)
            if match is not None:
                selected = selected.xs(match, axis=1, level=level)
                break
        if isinstance(selected.columns, pd.MultiIndex):
            selected.columns = selected.columns.get_level_values(field_level)
        return selected
    raise ValueError("unable to identify OHLCV fields in MultiIndex columns")


def _date_key(value: str | date | datetime | None, default: str) -> str:
    return default if value is None else pd.Timestamp(value).strftime("%Y%m%d")

