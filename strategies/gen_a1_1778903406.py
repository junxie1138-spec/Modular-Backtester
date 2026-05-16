from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticYieldParams:
    # Lookback over which net displacement and realized volatility are measured.
    lookback_bars: int = 20
    # Fixed number of bars to hold after entry (fixed-bar exit).
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[PlasticYieldParams]):
    """Trend-strength via the elastic/plastic deformation analogy.

    Net price displacement over N bars is compared against the elastic
    capacity of a random walk: daily_std * sqrt(N), expressed in price
    units. A yield ratio above 1.0 means the move is larger than
    diffusion can elastically explain - the structure has yielded
    plastically and the trend is treated as real. Entry fires on the
    upward cross of that yield point; the position is held for exactly
    hold_bars bars (fixed-bar exit, no signal-based exit).
    """

    strategy_id = "gen_a1_1778903406"

    @classmethod
    def params_type(cls):
        return PlasticYieldParams

    @staticmethod
    def warmup_bars(params: PlasticYieldParams) -> int:
        n = max(int(params.lookback_bars), 2)
        return n + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: PlasticYieldParams) -> pd.DataFrame:
        n = max(int(params.lookback_bars), 2)
        close = data["close"].astype(float)

        ret = close.pct_change()
        # Realized per-bar volatility over the lookback window.
        vol = ret.rolling(n).std()
        # Net displacement in price units over the same window.
        disp = close - close.shift(n)
        # Elastic capacity: random-walk excursion (return-space) -> price units.
        elastic = close.shift(n) * vol * np.sqrt(float(n))

        with np.errstate(divide="ignore", invalid="ignore"):
            yield_ratio = disp / elastic
        yield_ratio = yield_ratio.replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["yield_ratio"] = yield_ratio
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticYieldParams,
    ) -> SignalFrame:
        n = len(data)
        hold = max(int(params.hold_bars), 1)

        yr = indicators["yield_ratio"].to_numpy(dtype=float)
        # Yield point breached on the upside (NaN/inf -> False).
        above = np.isfinite(yr) & (yr > 1.0)

        # Entry triggers: upward cross of the elastic yield limit.
        trigger = np.zeros(n, dtype=bool)
        if n > 1:
            trigger[1:] = above[1:] & ~above[:-1]

        # Fixed-bar holding: each trigger holds a long for exactly `hold`
        # bars, then forces flat. No re-entry while a position is open.
        raw = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if trigger[i]:
                raw[i:min(i + hold, n)] = 1
                i += hold
            else:
                i += 1

        signal = pd.Series(raw, index=data.index, dtype=int)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        signal = signal.shift(1).fillna(0).astype(int)

        size = pd.Series(1.0, index=data.index, dtype=float)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
