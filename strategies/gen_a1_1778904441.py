from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RefractoryAccelParams:
    roc_period: int = 5
    z_window: int = 60
    entry_threshold: float = 2.0
    profit_target: float = 0.025
    time_stop: int = 2
    refractory_bars: int = 5
    size_min: float = 0.4
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[RefractoryAccelParams]):
    strategy_id = "gen_a1_1778904441"

    @classmethod
    def params_type(cls):
        return RefractoryAccelParams

    def warmup_bars(self, params):
        return int(params.roc_period + params.z_window + 5)

    def indicators(self, data, params):
        close = data["close"]
        roc_period = max(1, int(params.roc_period))
        zw = max(2, int(params.z_window))

        roc = close.pct_change(roc_period)
        accel = roc.diff(1)

        mean = accel.rolling(zw, min_periods=zw).mean()
        std = accel.rolling(zw, min_periods=zw).std()
        std = std.replace(0.0, np.nan)
        accel_z = (accel - mean) / std

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["accel"] = accel
        out["accel_z"] = accel_z
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        accel_z = indicators["accel_z"].to_numpy(dtype=float)
        n = len(close)

        sig = np.zeros(n, dtype=float)
        size = np.ones(n, dtype=float)

        thr = float(params.entry_threshold)
        if thr <= 0.0:
            thr = 1e-6
        pt = float(params.profit_target)
        ts = max(1, int(params.time_stop))
        refr = max(0, int(params.refractory_bars))
        smin = float(params.size_min)
        smax = float(params.size_max)
        if smin <= 0.0:
            smin = 0.01
        if smax < smin:
            smax = smin

        pos = 0
        entry_price = np.nan
        bars_held = 0
        refractory = 0
        cur_size = 1.0

        for i in range(n):
            z = accel_z[i]
            exited = False

            if pos != 0:
                bars_held += 1
                ret = (close[i] / entry_price - 1.0) * pos
                if ret >= pt or bars_held >= ts:
                    pos = 0
                    entry_price = np.nan
                    bars_held = 0
                    refractory = refr
                    exited = True

            if pos == 0 and not exited:
                if refractory > 0:
                    refractory -= 1
                elif np.isfinite(z) and abs(z) >= thr:
                    pos = 1 if z > 0 else -1
                    entry_price = close[i]
                    bars_held = 0
                    scale = abs(z) / thr
                    if scale < 1.0:
                        scale = 1.0
                    if scale > 2.0:
                        scale = 2.0
                    cur_size = smin + (smax - smin) * (scale - 1.0)
                    if cur_size < smin:
                        cur_size = smin
                    if cur_size > smax:
                        cur_size = smax

            sig[i] = pos
            size[i] = cur_size if pos != 0 else 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
