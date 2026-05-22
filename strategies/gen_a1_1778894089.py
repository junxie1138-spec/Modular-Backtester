from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionVolParams:
    median_window: int = 60
    atr_window: int = 20
    entry_k: float = 1.2
    spike_mult: float = 2.0
    refractory_bars: int = 5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[PositionVolParams]):
    strategy_id = "gen_a1_1778894089"

    @classmethod
    def params_type(cls) -> type[PositionVolParams]:
        return PositionVolParams

    def warmup_bars(self, params: PositionVolParams) -> int:
        return int(max(params.median_window, params.atr_window)) + 2

    def indicators(self, data: pd.DataFrame, params: PositionVolParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        median = close.rolling(
            params.median_window, min_periods=params.median_window
        ).median()

        # Volatility-normalized relative position vs the rolling median.
        atr_safe = atr.where(atr > 0.0)
        pos = (close - median) / atr_safe

        # A true-range spike relative to the prior ATR level.
        atr_prev = atr.shift(1)
        spike = (tr > params.spike_mult * atr_prev).fillna(False).astype(float)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["median"] = median
        out["pos"] = pos
        out["spike"] = spike
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PositionVolParams,
    ) -> SignalFrame:
        n = len(data)
        pos = indicators["pos"].to_numpy(dtype=float)
        spike = indicators["spike"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        position = 0
        refractory_until = -1
        k = float(params.entry_k)
        r = int(params.refractory_bars)

        for i in range(n):
            # A volatility spike opens a refractory window blocking new entries.
            if spike[i] > 0.0:
                refractory_until = i + r

            o = pos[i]
            if not np.isfinite(o):
                raw[i] = position
                continue

            in_refractory = i <= refractory_until

            if position == 0:
                # Symmetric entry: positional lead exceeds +k ATRs.
                if o > k and not in_refractory:
                    position = 1
            else:
                # Mirror (signal-reversal) exit: lead flips to below -k ATRs.
                if o < -k:
                    position = 0

            raw[i] = position

        df = pd.DataFrame(index=data.index)
        signal = pd.Series(raw, index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = float(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
