from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rv_window: int = 10
    baseline_window: int = 60
    trend_window: int = 40
    leak: float = 0.15
    capacity: float = 0.06
    profit_target: float = 0.04
    time_stop: int = 9
    vol_target: float = 0.012
    size_gain: float = 3.0
    max_size: float = 1.5
    rv_floor: float = 0.003


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778888364"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        a = int(params.rv_window) + int(params.baseline_window) + 1
        b = int(params.trend_window) + 1
        return int(max(a, b))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        rv = ret.rolling(int(params.rv_window)).std()
        baseline = rv.rolling(int(params.baseline_window)).mean()
        # arrival: positive when realized vol sits below its baseline (vol suppressed)
        arrival = (baseline - rv).fillna(0.0).to_numpy(dtype=float)

        n = len(close)
        leak = float(params.leak)
        queue = np.zeros(n, dtype=float)
        for i in range(1, n):
            queue[i] = max(0.0, (1.0 - leak) * queue[i - 1] + arrival[i])

        trend = close.pct_change(int(params.trend_window))

        out = pd.DataFrame(index=data.index)
        out["rv"] = rv
        out["baseline"] = baseline
        out["queue"] = queue
        out["trend"] = trend
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        rv = indicators["rv"].to_numpy(dtype=float)
        queue = indicators["queue"].to_numpy(dtype=float)
        trend = indicators["trend"].to_numpy(dtype=float)

        n = len(close)
        sig = np.zeros(n, dtype=float)
        size = np.ones(n, dtype=float)

        capacity = float(params.capacity)
        profit_target = float(params.profit_target)
        time_stop = int(params.time_stop)
        vol_target = float(params.vol_target)
        size_gain = float(params.size_gain)
        max_size = float(params.max_size)
        rv_floor = float(params.rv_floor)

        pos = 0
        bars_held = 0
        entry_price = 0.0
        held_size = 1.0

        for i in range(n):
            if pos == 0:
                q = queue[i]
                tr = trend[i]
                if np.isfinite(q) and np.isfinite(tr) and q > capacity and tr != 0.0:
                    direction = 1 if tr > 0.0 else -1
                    overflow_norm = (q - capacity) / capacity
                    strength = 1.0 + size_gain * overflow_norm
                    rv_i = rv[i]
                    rv_eff = rv_i if (np.isfinite(rv_i) and rv_i > rv_floor) else rv_floor
                    raw = (vol_target / rv_eff) * strength
                    held_size = float(np.clip(raw, 0.1, max_size))
                    pos = direction
                    bars_held = 0
                    entry_price = close[i]
                    sig[i] = float(direction)
                    size[i] = held_size
                else:
                    sig[i] = 0.0
                    size[i] = 1.0
            else:
                bars_held += 1
                pnl = ((close[i] - entry_price) / entry_price) * pos
                if pnl >= profit_target or bars_held >= time_stop:
                    sig[i] = 0.0
                    size[i] = 1.0
                    pos = 0
                    bars_held = 0
                    entry_price = 0.0
                else:
                    sig[i] = float(pos)
                    size[i] = held_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).clip(lower=0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
