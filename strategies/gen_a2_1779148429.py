from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    anchor_window: int = 20
    range_window: int = 20
    entry_thresh: float = 1.5
    range_expansion_mult: float = 1.0
    ma_window: int = 200
    exit_level: float = 0.0
    size_base: float = 1.0
    size_max: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779148429"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.anchor_window, params.range_window, params.ma_window)) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        p = params
        high = data["high"]
        low = data["low"]
        close = data["close"]

        mid = (high + low) / 2.0
        anchor = mid.rolling(p.anchor_window, min_periods=p.anchor_window).mean()

        rng = (high - low).clip(lower=0.0)
        avg_rng = rng.rolling(p.range_window, min_periods=p.range_window).mean()
        avg_rng = avg_rng.where(avg_rng > 0.0)

        disp = (close - anchor) / avg_rng

        ma = close.rolling(p.ma_window, min_periods=p.ma_window).mean()

        range_expanding = (rng > (avg_rng * p.range_expansion_mult)).astype(float)
        above_ma = (close > ma).astype(float)
        below_ma = (close < ma).astype(float)

        out = pd.DataFrame(index=data.index)
        out["disp"] = disp
        out["range_expanding"] = range_expanding
        out["above_ma"] = above_ma
        out["below_ma"] = below_ma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        p = params
        n = len(data)

        disp = indicators["disp"].to_numpy(dtype=float)
        rng_exp = indicators["range_expanding"].to_numpy(dtype=float)
        above_ma = indicators["above_ma"].to_numpy(dtype=float)
        below_ma = indicators["below_ma"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        entry_thresh = max(float(p.entry_thresh), 1e-9)
        exit_level = float(p.exit_level)

        pos = 0
        cur_size = 1.0
        for i in range(n):
            d = disp[i]
            if not np.isfinite(d):
                signal[i] = pos
                size[i] = cur_size
                continue

            if pos == 0:
                expanding = rng_exp[i] >= 1.0
                if expanding and d <= -entry_thresh and above_ma[i] >= 1.0:
                    pos = 1
                    cur_size = min(
                        max(p.size_base * abs(d) / entry_thresh, p.size_base),
                        p.size_max,
                    )
                elif expanding and d >= entry_thresh and below_ma[i] >= 1.0:
                    pos = -1
                    cur_size = min(
                        max(p.size_base * abs(d) / entry_thresh, p.size_base),
                        p.size_max,
                    )
            elif pos == 1:
                if d >= exit_level:
                    pos = 0
                    cur_size = 1.0
            elif pos == -1:
                if d <= -exit_level:
                    pos = 0
                    cur_size = 1.0

            signal[i] = pos
            size[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["size"] = df["size"].clip(lower=1e-6)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
