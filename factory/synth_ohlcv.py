from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n_bars: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a deterministic ~n_bars OHLCV frame with realistic constraints.

    Constraints: high >= max(open, close) >= min(open, close) >= low; volume > 0.
    Index is daily business dates. Prices float around 100 via a random walk
    plus a low-amplitude sinusoid so percentile/range-style strategies have
    something to react to.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_bars)
    trend = np.linspace(0.0, 5.0, n_bars)
    wave = 3.0 * np.sin(np.linspace(0.0, 6 * np.pi, n_bars))
    noise = rng.normal(0.0, 0.6, n_bars).cumsum()
    close = 100.0 + trend + wave + noise

    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.2, n_bars - 1)

    intrabar = rng.uniform(0.3, 1.5, n_bars)
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar
    volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df
