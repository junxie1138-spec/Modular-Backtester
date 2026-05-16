from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenParams:
    peak_lookback: int = 60
    dd_threshold: float = 0.05
    absorb_threshold: float = 0.8
    spike_gap_thresh: float = 0.02
    refractory_bars: int = 3
    profit_target: float = 0.02
    time_stop: int = 2
    base_size: float = 0.6
    dd_size_mult: float = 4.0


class GeneratedStrategy(BaseStrategy[GenParams]):
    strategy_id = "gen_a1_1778894398"

    @classmethod
    def params_type(cls):
        return GenParams

    def warmup_bars(self, params: GenParams) -> int:
        return int(params.peak_lookback) + 2

    def indicators(self, data: pd.DataFrame, params: GenParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)

        prior_close = data["close"].shift(1)
        gap = prior_close - data["open"]
        down = gap > 0.0
        safe_gap = gap.where(down)
        recovery = data["close"] - data["open"]
        recovery_ratio = recovery / safe_gap
        absorbed = (down & (recovery_ratio >= p.absorb_threshold)).astype(float)

        lookback = max(int(p.peak_lookback), 2)
        roll_peak = data["close"].rolling(lookback, min_periods=lookback).max()
        drawdown = data["close"] / roll_peak - 1.0

        gap_pct = data["open"] / prior_close - 1.0
        is_spike = (gap_pct.abs() >= p.spike_gap_thresh).astype(float)

        dd_depth = (-drawdown).clip(lower=0.0).fillna(0.0)
        size = (p.base_size + p.dd_size_mult * dd_depth).clip(lower=0.1, upper=1.0)

        out["absorbed_down_gap"] = absorbed.fillna(0.0)
        out["drawdown"] = drawdown
        out["is_spike"] = is_spike.fillna(0.0)
        out["size"] = size.fillna(0.1)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        absorbed = indicators["absorbed_down_gap"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        is_spike = indicators["is_spike"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0
        last_spike = -10**9
        time_stop = max(int(p.time_stop), 1)
        refractory = max(int(p.refractory_bars), 0)
        dd_trigger = -abs(float(p.dd_threshold))

        for i in range(n):
            if is_spike[i] >= 1.0:
                last_spike = i
            if position == 0:
                in_refractory = (i - last_spike) <= refractory
                two_bar = (
                    i >= 1
                    and absorbed[i] >= 1.0
                    and absorbed[i - 1] >= 1.0
                )
                dd_ok = dd[i] <= dd_trigger
                if two_bar and dd_ok and not in_refractory:
                    raw[i] = 1
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                else:
                    raw[i] = 0
            else:
                bars_held += 1
                hit_pt = (
                    entry_price > 0.0
                    and close[i] >= entry_price * (1.0 + p.profit_target)
                )
                hit_time = bars_held >= time_stop
                if hit_pt or hit_time:
                    raw[i] = 0
                    position = 0
                else:
                    raw[i] = 1

        out = pd.DataFrame(index=data.index)
        out["signal"] = raw
        out["signal"] = out["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].to_numpy(dtype=float)
        size = np.where(np.isfinite(size) & (size > 0.0), size, 0.1)
        out["size"] = size
        return SignalFrame(data=out, signal_column="signal", size_column="size")
