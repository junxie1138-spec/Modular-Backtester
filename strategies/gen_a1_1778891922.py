from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilParams:
    coil_window: int = 10
    coil_min: int = 4
    breakout_window: int = 10
    atr_window: int = 14
    ma_window: int = 200
    spike_mult: float = 2.5
    refractory: int = 5
    trail_k: float = 2.5


class GeneratedStrategy(BaseStrategy[CoilParams]):
    strategy_id = "gen_a1_1778891922"

    @classmethod
    def params_type(cls) -> type[CoilParams]:
        return CoilParams

    def warmup_bars(self, params: CoilParams) -> int:
        longest = max(
            params.ma_window,
            params.atr_window,
            params.breakout_window,
            params.coil_window,
        )
        return int(longest) + 2

    def indicators(self, data: pd.DataFrame, params: CoilParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        # True range and ATR (volatility primitive).
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        # Inside-bar nesting: current bar fully contained in the prior bar.
        inside = ((high <= prev_high) & (low >= prev_low)).astype(float)
        coil_count = inside.rolling(params.coil_window).sum()

        # Highest close over the PRIOR breakout_window bars (excludes current).
        breakout_high = close.rolling(params.breakout_window).max().shift(1)

        # 200-day regime filter.
        ma_long = close.rolling(params.ma_window).mean()

        # Volatility spike: an outsized true range relative to ATR.
        spike = (tr > (params.spike_mult * atr)).astype(float)
        spike = spike.where(atr.notna(), 0.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["coil_count"] = coil_count
        out["breakout_high"] = breakout_high
        out["ma_long"] = ma_long
        out["spike"] = spike
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        coil_count = indicators["coil_count"].to_numpy(dtype=float)
        breakout_high = indicators["breakout_high"].to_numpy(dtype=float)
        ma_long = indicators["ma_long"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        hwm = 0.0  # highest close since entry (ratchets up only)
        bars_since_spike = 10 ** 6

        for i in range(n):
            if spike[i] >= 1.0:
                bars_since_spike = 0
            else:
                bars_since_spike += 1

            atr_i = atr[i]
            atr_ok = not np.isnan(atr_i)

            if in_pos:
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.trail_k * atr_i if atr_ok else -np.inf
                if atr_ok and close[i] <= stop:
                    in_pos = False
                    hwm = 0.0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                cc = coil_count[i]
                bh = breakout_high[i]
                ma = ma_long[i]
                ready = (
                    atr_ok
                    and not np.isnan(cc)
                    and not np.isnan(bh)
                    and not np.isnan(ma)
                )
                entry = (
                    ready
                    and cc >= params.coil_min
                    and close[i] > bh
                    and close[i] > ma
                    and bars_since_spike >= params.refractory
                )
                if entry:
                    in_pos = True
                    hwm = close[i]
                    signal[i] = 1
                else:
                    signal[i] = 0

        out = pd.DataFrame(index=data.index)
        out["signal"] = signal
        out["size"] = size
        out["signal"] = out["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
