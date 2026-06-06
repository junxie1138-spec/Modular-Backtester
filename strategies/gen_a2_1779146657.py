from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 20
    queue_window: int = 30
    high_band: float = 0.80
    low_band: float = 0.20
    capacity_frac: float = 0.50
    max_hold: int = 18
    profit_target: float = 0.06


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779146657"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.rank_window + params.queue_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        rank_pct = close.rolling(params.rank_window).rank(pct=True)

        high_ext = (rank_pct >= params.high_band).astype(float)
        low_ext = (rank_pct <= params.low_band).astype(float)
        # keep warmup bars as NaN so the queue counts only valid extremes
        high_ext[rank_pct.isna()] = np.nan
        low_ext[rank_pct.isna()] = np.nan

        count_high = high_ext.rolling(params.queue_window).sum()
        count_low = low_ext.rolling(params.queue_window).sum()

        ind = pd.DataFrame(index=data.index)
        ind["rank_pct"] = rank_pct
        ind["count_high"] = count_high
        ind["count_low"] = count_low
        return ind

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        rank_pct = indicators["rank_pct"].to_numpy(dtype=float)
        count_high = indicators["count_high"].to_numpy(dtype=float)
        count_low = indicators["count_low"].to_numpy(dtype=float)

        capacity = float(params.capacity_frac) * float(params.queue_window)

        # per-bar directional candidate: momentum below capacity, reversal on overflow
        raw = np.zeros(n, dtype=float)
        for i in range(n):
            rp = rank_pct[i]
            ch = count_high[i]
            cl = count_low[i]
            if np.isnan(rp) or np.isnan(ch) or np.isnan(cl):
                raw[i] = 0.0
                continue
            if rp >= params.high_band:
                raw[i] = 1.0 if ch <= capacity else -1.0
            elif rp <= params.low_band:
                raw[i] = -1.0 if cl <= capacity else 1.0
            else:
                raw[i] = 0.0

        signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(n):
            if position != 0:
                bars_held += 1
                ret = (close[i] / entry_price - 1.0) * position
                hit_target = ret >= params.profit_target
                hit_time = bars_held >= params.max_hold
                if hit_target or hit_time:
                    position = 0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = position
                continue
            # flat: require two-bar confirmation of the same directional candidate
            if i >= 1 and raw[i] != 0.0 and raw[i] == raw[i - 1]:
                position = int(raw[i])
                entry_price = close[i]
                bars_held = 0
                signal[i] = position
            else:
                signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
