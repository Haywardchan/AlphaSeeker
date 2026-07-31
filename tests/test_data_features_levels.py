import numpy as np
import pandas as pd
import pytest

from alpha_seeker.config import AlphaConfig
from alpha_seeker.data import YFinanceProvider, normalize_ohlcv, validate_ohlcv
from alpha_seeker.features import atr, make_features
from alpha_seeker.levels import causal_level_frame, estimate_levels


def test_normalize_yahoo_multiindex_and_dates(bars):
    raw = bars.iloc[:3].copy()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["BRK-B"]])
    raw.index = raw.index.tz_localize("US/Eastern")
    result = normalize_ohlcv(raw)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.tz is None
    assert result.dtypes.eq("float64").all()


def test_invalid_ohlc_is_rejected(bars):
    bad = bars.iloc[:3].copy()
    bad.iloc[0, bad.columns.get_loc("High")] = 1
    with pytest.raises(ValueError, match="High"):
        validate_ohlcv(bad)


def test_features_are_causal(bars):
    full = make_features(bars)
    prefix = make_features(bars.iloc[:220])
    pd.testing.assert_series_equal(full.loc[prefix.index[-1]], prefix.iloc[-1])


def test_atr_and_level_fallback_are_finite(bars):
    values = atr(bars, 14)
    assert values.iloc[13:].notna().all()
    levels = causal_level_frame(bars, lookback=30)
    usable = levels.iloc[20:]
    assert np.isfinite(usable[["support", "resistance"]]).all().all()
    assert (usable["support"] < bars.loc[usable.index, "Close"]).all()
    assert (usable["resistance"] > bars.loc[usable.index, "Close"]).all()
    support, resistance = estimate_levels(bars, lookback=30)
    assert support.lower <= support.price <= support.upper
    assert resistance.lower <= resistance.price <= resistance.upper
    assert levels["source"].str.startswith(("kde", "atr", "unavailable")).all()


def test_provider_uses_validated_disk_cache(bars, tmp_path):
    calls = 0

    def downloader(**_kwargs):
        nonlocal calls
        calls += 1
        return bars.iloc[:20]

    config = AlphaConfig(cache_dir=tmp_path, cache_ttl_seconds=None)
    provider = YFinanceProvider(config, downloader=downloader)
    first = provider.get(start="2020-01-01", end="2020-02-01")
    second = provider.get(start="2020-01-01", end="2020-02-01")
    assert calls == 1
    pd.testing.assert_frame_equal(first, second)
