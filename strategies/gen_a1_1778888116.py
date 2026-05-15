from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 50
    trail_atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Relative-position acceleration entry with a breakeven-then-trail exit.

    The in-range position r = (close - rolling_low) / (rolling_high - rolling_low)
    is treated like the infected fraction of an SI epidemic curve. Entry fires
    when r sits in an early, non-saturated band and is rising with positive
    curvature (positive second difference). The position is then managed by a
    path-dependent stop that only ever ratchets up.
    """

    strategy_id = "gen_a1_1778888116"

    _ATR_PERIOD = 14
    _BREAKEVEN_PCT = 0.03
    _R_LOW = 0.15
    _R_HIGH = 0.75

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        w = int(max(2, params.window))
        # rolling min/max needs w bars; diff().diff() consumes 2 more.
        return int(max(w + 3, self._ATR_PERIOD + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        w = int(max(2, params.window))

        roll_min = low.rolling(w, min_periods=w).min()
        roll_max = high.rolling(w, min_periods=w).max()
        span = (roll_max - roll_min).replace(0.0, np.nan)
        r = ((close - roll_min) / span).clip(lower=0.0, upper=1.0)

        velocity = r.diff()
        accel = velocity.diff()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_PERIOD, min_periods=self._ATR_PERIOD).mean()

        entry = (
            (r >= self._R_LOW)
            & (r <= self._R_HIGH)
            & (velocity > 0.0)
            & (accel > 0.0)
        ).fillna(False)

        ind = pd.DataFrame(index=data.index)
        ind["r"] = r
        ind["velocity"] = velocity
        ind["accel"] = accel
        ind["atr"] = atr
        ind["entry"] = entry.astype(bool)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=bool)

        n = len(close)
        signal = np.zeros(n, dtype=np.int64)
        k = float(params.trail_atr_mult)
        be_pct = self._BREAKEVEN_PCT

        position = 0
        entry_price = 0.0
        stop = 0.0
        breakeven_done = False

        for i in range(n):
            if position == 0:
                if entry[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    stop = entry_price - k * atr[i]
                    breakeven_done = False
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                a = atr[i] if np.isfinite(atr[i]) else 0.0
                # Breakeven: once price has run +X%, lock the stop at entry.
                if (not breakeven_done) and high[i] >= entry_price * (1.0 + be_pct):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_done = True
                # Trail: stop only ratchets up, never down.
                trail = close[i] - k * a
                if trail > stop:
                    stop = trail
                # Exit if the bar trades through the stop.
                if low[i] <= stop:
                    position = 0
                    entry_price = 0.0
                    stop = 0.0
                    breakeven_done = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
