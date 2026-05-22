from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    range_window: int = 40
    vol_window: int = 20
    band: float = 0.20
    vol_mult: float = 1.3
    size_floor: float = 0.5
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[TideParams]):
    """Volume-confirmed range-position stop-and-reverse.

    The 'tide level' is the fraction of the rolling high-low range the close
    currently occupies. Long and short entries are exact mirror images about
    mid-range (0.5): a long fires when the tide crosses up through 0.5 + band
    on confirming volume, a short fires when it crosses down through
    0.5 - band on confirming volume. The only exit is the opposite (symmetric)
    entry firing - a pure signal-reversal exit.
    """

    strategy_id = "gen_a1_1778883984"

    @classmethod
    def params_type(cls) -> type[TideParams]:
        return TideParams

    @staticmethod
    def warmup_bars(params: TideParams) -> int:
        return int(max(params.range_window, params.vol_window)) + 2

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rw = int(params.range_window)
        vw = int(params.vol_window)

        highest = high.rolling(rw, min_periods=rw).max()
        lowest = low.rolling(rw, min_periods=rw).min()
        rng = highest - lowest
        valid = rng > 0

        # Range-position fraction: 0.0 = low tide, 1.0 = high tide.
        pos = (close - lowest) / rng.where(valid)
        pos = pos.where(valid, 0.5)
        pos = pos.clip(lower=0.0, upper=1.0)

        vol_avg = volume.rolling(vw, min_periods=vw).mean()
        vol_ratio = volume / vol_avg.where(vol_avg > 0)

        out = pd.DataFrame(index=data.index)
        out["pos"] = pos
        out["vol_ratio"] = vol_ratio
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideParams,
    ) -> SignalFrame:
        pos = indicators["pos"]
        vol_ratio = indicators["vol_ratio"]

        band = float(params.band)
        upper = 0.5 + band
        lower = 0.5 - band

        pos_prev = pos.shift(1)
        vol_ok = (vol_ratio >= float(params.vol_mult)).fillna(False)

        # Symmetric entry rule: long crosses up through upper band,
        # short crosses down through the mirror-image lower band.
        long_entry = (pos >= upper) & (pos_prev < upper) & vol_ok
        short_entry = (pos <= lower) & (pos_prev > lower) & vol_ok

        long_arr = long_entry.fillna(False).to_numpy()
        short_arr = short_entry.fillna(False).to_numpy()

        # Stop-and-reverse state machine: the only way out of a position is
        # the symmetric opposite entry firing (signal-reversal exit).
        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        state = 0
        for i in range(n):
            if long_arr[i] and state <= 0:
                state = 1
            elif short_arr[i] and state >= 0:
                state = -1
            raw[i] = state

        df = pd.DataFrame(index=data.index)
        signal = pd.Series(raw, index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = signal.shift(1).fillna(0).astype(int)

        # Conviction sizing: scale with the confirming volume ratio.
        size = vol_ratio.clip(
            lower=float(params.size_floor), upper=float(params.size_cap)
        )
        size = size.fillna(1.0)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
