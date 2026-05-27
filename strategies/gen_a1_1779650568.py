from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarVolShockParams:
    short_vol_window: int = 5
    long_vol_window: int = 8
    vol_shock_threshold: float = 1.10
    hold_bars: int = 15
    tom_days: int = 3
    mid_month_start: int = 10
    mid_month_end: int = 20


class GeneratedStrategy(BaseStrategy[CalendarVolShockParams]):
    strategy_id = "gen_a1_1779650568"

    @classmethod
    def params_type(cls):
        return CalendarVolShockParams

    @classmethod
    def warmup_bars(cls, params: CalendarVolShockParams) -> int:
        return int(max(params.short_vol_window, params.long_vol_window)) + 1

    def indicators(self, data: pd.DataFrame, params: CalendarVolShockParams) -> pd.DataFrame:
        close = data["close"]
        rets = close.pct_change()

        sw = int(params.short_vol_window)
        lw = int(params.long_vol_window)
        vol_s = rets.rolling(sw, min_periods=sw).std()
        vol_l = rets.rolling(lw, min_periods=lw).std()
        vol_ratio = vol_s / vol_l.replace(0.0, np.nan)

        idx = data.index
        ym_codes = pd.Series(idx.year * 12 + idx.month, index=idx)
        pos_from_start = ym_codes.groupby(ym_codes).cumcount().astype(np.int64)
        month_size = ym_codes.groupby(ym_codes).transform("size").astype(np.int64)
        pos_from_end = (month_size - 1 - pos_from_start).astype(np.int64)

        tom_days = int(params.tom_days)
        in_tom = (pos_from_start < tom_days) | (pos_from_end < tom_days)
        in_mid = (pos_from_start >= int(params.mid_month_start)) & (pos_from_start <= int(params.mid_month_end))

        shock = (vol_ratio > float(params.vol_shock_threshold)).fillna(False)

        raw_signal = pd.Series(0, index=idx, dtype=np.int64)
        long_mask = (shock & in_tom).fillna(False)
        short_mask = (shock & in_mid & (~in_tom)).fillna(False)
        raw_signal[long_mask] = 1
        raw_signal[short_mask] = -1

        out = pd.DataFrame(index=idx)
        out["rets"] = rets
        out["vol_short"] = vol_s
        out["vol_long"] = vol_l
        out["vol_ratio"] = vol_ratio
        out["pos_from_start"] = pos_from_start.astype(float)
        out["pos_from_end"] = pos_from_end.astype(float)
        out["in_tom"] = in_tom.astype(np.int64)
        out["in_mid"] = in_mid.astype(np.int64)
        out["shock"] = shock.astype(np.int64)
        out["raw_signal"] = raw_signal
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CalendarVolShockParams) -> SignalFrame:
        n = len(data)
        raw = indicators["raw_signal"].to_numpy().astype(np.int64)
        sig = np.zeros(n, dtype=np.int64)
        hold_left = 0
        current_dir = 0
        hold_bars = int(params.hold_bars)
        for i in range(n):
            if hold_left > 0:
                sig[i] = current_dir
                hold_left -= 1
                continue
            cand = int(raw[i])
            if cand != 0:
                current_dir = cand
                sig[i] = current_dir
                hold_left = hold_bars - 1
            else:
                current_dir = 0
                sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(1.0, index=data.index, dtype=float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
