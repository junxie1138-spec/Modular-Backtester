from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRetentionParams:
    ma_window: int = 200
    dd_window: int = 60
    arm_lookback: int = 25
    retention_window: int = 30
    dd_arm: float = 0.07
    ru_arm: float = 0.07
    ret_thresh: float = 0.60
    profit_target: float = 0.06
    max_hold: int = 18
    size_k: float = 4.0
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[GapRetentionParams]):
    strategy_id = "gen_a1_1778893974"

    @classmethod
    def params_type(cls):
        return GapRetentionParams

    @staticmethod
    def warmup_bars(params: GapRetentionParams) -> int:
        return int(params.ma_window + params.retention_window + params.arm_lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapRetentionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        prior_close = close.shift(1)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        gap = open_ - prior_close
        up_gap = gap > 0.0
        down_gap = gap < 0.0
        # A gap is 'retained' (plastic) when the close holds the gap's direction
        # relative to the prior close; otherwise it was filled back (elastic).
        up_held = up_gap & (close > prior_close)
        down_held = down_gap & (close < prior_close)

        w = max(2, int(params.retention_window))
        up_gap_n = up_gap.rolling(w, min_periods=1).sum()
        down_gap_n = down_gap.rolling(w, min_periods=1).sum()
        up_held_n = up_held.rolling(w, min_periods=1).sum()
        down_held_n = down_held.rolling(w, min_periods=1).sum()

        up_ret = up_held_n / up_gap_n.replace(0.0, np.nan)
        down_ret = down_held_n / down_gap_n.replace(0.0, np.nan)

        roll_max = close.rolling(params.dd_window, min_periods=1).max()
        roll_min = close.rolling(params.dd_window, min_periods=1).min()
        dd = close / roll_max - 1.0
        ru = close / roll_min - 1.0

        al = max(1, int(params.arm_lookback))
        dd_min = dd.rolling(al, min_periods=1).min()
        ru_max = ru.rolling(al, min_periods=1).max()
        trough = (-dd_min).clip(lower=0.0)
        peak = ru_max.clip(lower=0.0)

        ind = pd.DataFrame(index=data.index)
        ind["ma"] = ma
        ind["up_ret"] = up_ret.fillna(0.0)
        ind["down_ret"] = down_ret.fillna(0.0)
        ind["dd_min"] = dd_min.fillna(0.0)
        ind["ru_max"] = ru_max.fillna(0.0)
        ind["trough"] = trough.fillna(0.0)
        ind["peak"] = peak.fillna(0.0)
        return ind

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRetentionParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        up_ret = indicators["up_ret"].to_numpy(dtype=float)
        down_ret = indicators["down_ret"].to_numpy(dtype=float)
        dd_min = indicators["dd_min"].to_numpy(dtype=float)
        ru_max = indicators["ru_max"].to_numpy(dtype=float)
        trough = indicators["trough"].to_numpy(dtype=float)
        peak = indicators["peak"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        pos = 0
        entry_price = 0.0
        entry_size = 1.0
        bars_held = 0
        thr = float(params.ret_thresh)
        pt = float(params.profit_target)
        max_hold = max(1, int(params.max_hold))
        cap = max(1.0, float(params.size_cap))

        for i in range(1, n):
            ma_ok = np.isfinite(ma[i])
            bull = ma_ok and close[i] > ma[i]
            bear = ma_ok and close[i] < ma[i]

            armed_long = dd_min[i] <= -params.dd_arm
            armed_short = ru_max[i] >= params.ru_arm
            cross_long = (up_ret[i] > thr) and (up_ret[i - 1] <= thr)
            cross_short = (down_ret[i] > thr) and (down_ret[i - 1] <= thr)

            if pos == 0:
                if armed_long and cross_long and bull:
                    pos = 1
                    entry_price = close[i]
                    bars_held = 0
                    entry_size = min(cap, 1.0 + params.size_k * max(0.0, trough[i]))
                    signal[i] = 1
                    size[i] = entry_size
                elif armed_short and cross_short and bear:
                    pos = -1
                    entry_price = close[i]
                    bars_held = 0
                    entry_size = min(cap, 1.0 + params.size_k * max(0.0, peak[i]))
                    signal[i] = -1
                    size[i] = entry_size
            elif pos == 1:
                bars_held += 1
                hit_pt = entry_price > 0.0 and close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    pos = 0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                bars_held += 1
                hit_pt = entry_price > 0.0 and close[i] <= entry_price * (1.0 - pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    pos = 0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = -1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = np.where(np.isfinite(size) & (size > 0.0), size, 1.0)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
