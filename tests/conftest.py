import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def bars() -> pd.DataFrame:
    size = 320
    index = pd.bdate_range("2020-01-02", periods=size)
    trend = 200 + np.arange(size) * 0.12
    close = trend + 4 * np.sin(np.arange(size) / 9)
    open_ = close * (1 + 0.002 * np.cos(np.arange(size) / 5))
    high = np.maximum(open_, close) + 2
    low = np.minimum(open_, close) - 2
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000 + np.arange(size) * 100,
        },
        index=index,
    )
