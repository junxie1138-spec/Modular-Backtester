from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 500,
    seed: int = 0,
    start: str = "2020-01-02",
    start_price: float = 100.0,
    drift: float = 0.0003,
    vol: float = 0.012,
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-style OHLCV series for tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    close = start_price * np.exp(np.cumsum(log_returns))

    # open ~ prev close with small gap
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.001, n - 1))

    intrabar_range = np.abs(rng.normal(0.0, vol, n)) * close
    high = np.maximum(open_, close) + intrabar_range
    low = np.minimum(open_, close) - intrabar_range
    low = np.clip(low, 0.01, None)
    volume = rng.integers(500_000, 5_000_000, size=n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
