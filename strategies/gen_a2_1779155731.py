from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    z_window: int = 20
    z_thresh: float = 1.0
    def_window: int = 15
    yield_mult: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779155731"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.ma_window + max(params.z_window, params.def_window) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        # Price displacement from the MA, expressed as a rolling z-score.
        dist = close - ma
        dist_mean = dist.rolling(params.z_window, min_periods=params.z_window).mean()
        dist_std = dist.rolling(params.z_window, min_periods=params.z_window).std()
        z = (dist - dist_mean) / dist_std.replace(0.0, np.nan)

        # Elastic vs plastic deformation test applied to the MA itself.
        # Net MA displacement over def_window vs the displacement a random walk
        # of equal-variance steps would produce (the elastic / yield limit).
        ma_step = ma.diff()
        ma_noise = ma_step.rolling(params.def_window, min_periods=params.def_window).std()
        elastic_limit = ma_noise * np.sqrt(float(params.def_window))
        ma_disp = ma - ma.shift(params.def_window)
        plastic = (ma_disp > params.yield_mult * elastic_limit).astype(float)

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["plastic"] = plastic
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        z = indicators["z"]
        plastic = indicators["plastic"]

        # Breakout bar: price extended above MA AND MA has plastically migrated.
        breakout = (z > params.z_thresh) & (plastic > 0.5)
        breakout = breakout.fillna(False)

        # Hard twist: two-bar confirmation -- the breakout state must hold on
        # two consecutive bars before an entry is permitted.
        confirmed = (breakout & breakout.shift(1).fillna(False)).to_numpy()

        n = len(data)
        sig = np.zeros(n, dtype=np.int64)
        hold = max(1, int(params.hold_bars))

        # Fixed-bar exit: hold exactly hold_bars bars, no signal-based exit.
        i = 0
        while i < n:
            if confirmed[i]:
                end = min(i + hold, n)
                sig[i:end] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
