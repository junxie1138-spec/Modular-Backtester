from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    # Only two tunable params by design (hard twist).
    window: int = 20      # lookback for the return-sign consensus
    band: float = 0.30    # upper Schmitt threshold; consensus in [-1, 1]


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778905254"

    # Fixed mechanics kept off the tunable surface (<=2 tunable params).
    _ATR_WINDOW = 14
    _BREAKEVEN_PCT = 0.03   # price must reach +3% before stop jumps to breakeven
    _TRAIL_K = 2.5          # trailing stop distance = TRAIL_K * ATR
    _REARM_LEVEL = 0.0      # consensus must fall to <= 0 to re-arm the latch

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        # consensus uses diff() then a rolling window -> window + 1.
        return int(max(int(params.window), cls._ATR_WINDOW)) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Close-to-close return sign -> rolling unanimity in [-1, 1].
        ret = close.diff()
        ret_sign = np.sign(ret)
        win = max(int(params.window), 1)
        consensus = ret_sign.rolling(win, min_periods=win).mean()

        # ATR for the breakeven-then-trail exit.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_WINDOW, min_periods=self._ATR_WINDOW).mean()

        out = pd.DataFrame(index=data.index)
        out["consensus"] = consensus
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        consensus = indicators["consensus"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        upper = float(params.band)
        lower = float(self._REARM_LEVEL)

        raw = np.zeros(n, dtype=int)

        state_high = False    # Schmitt-trigger latch state
        in_pos = False
        entry_price = 0.0
        hwm = 0.0             # high-water mark since entry
        stop = 0.0
        be_armed = False      # has the stop been lifted to breakeven yet

        for i in range(n):
            c = consensus[i]

            # --- hysteresis latch: dual-threshold Schmitt trigger ---
            prev_state_high = state_high
            if not np.isnan(c):
                if c >= upper:
                    state_high = True
                elif c <= lower:
                    state_high = False
            entry_event = (not prev_state_high) and state_high

            if in_pos:
                a = atr[i]
                if np.isnan(a):
                    a = 0.0
                if high[i] > hwm:
                    hwm = high[i]

                # breakeven-then-trail: stop only ever moves up.
                if (not be_armed) and high[i] >= entry_price * (1.0 + self._BREAKEVEN_PCT):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price
                trail = hwm - self._TRAIL_K * a
                if trail > stop:
                    stop = trail

                if low[i] <= stop:
                    in_pos = False
                    be_armed = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                # Fresh long only on a LOW->HIGH latch transition.
                if entry_event:
                    in_pos = True
                    be_armed = False
                    entry_price = close[i]
                    hwm = high[i]
                    a = atr[i]
                    if np.isnan(a):
                        a = 0.0
                    stop = entry_price - self._TRAIL_K * a
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
