from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_window: int = 5
    regime_window: int = 4
    trend_window: int = 8
    hold_bars: int = 7
    accel_thresh: float = 0.0
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778909322"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # vol needs pct_change(1) + vol_window; vol_ma adds regime_window;
        # vol_accel adds 2 diffs; trend_sma needs trend_window. All <= 10.
        return 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        vol = ret.rolling(params.vol_window).std()
        vol_ma = vol.rolling(params.regime_window).mean()
        vol_diff = vol.diff()
        vol_accel = vol_diff.diff()
        trend_sma = close.rolling(params.trend_window).mean()

        out = pd.DataFrame(index=data.index)
        out["vol"] = vol
        out["vol_ma"] = vol_ma
        out["vol_diff"] = vol_diff
        out["vol_accel"] = vol_accel
        out["trend_sma"] = trend_sma
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"]
        vol = indicators["vol"]
        vol_ma = indicators["vol_ma"]
        vol_diff = indicators["vol_diff"]
        vol_accel = indicators["vol_accel"]
        trend_sma = indicators["trend_sma"]

        # Predator population large: volatility above its own regime average.
        elevated = vol > vol_ma
        # Predator rolling over: vol curvature negative AND vol falling now.
        rolling_over = (vol_accel < params.accel_thresh) & (vol_diff < 0.0)
        # Prey alive: price above its short trend baseline.
        uptrend = close > trend_sma

        entry = (elevated & rolling_over & uptrend).fillna(False).to_numpy()

        n = len(data)
        sig = np.zeros(n, dtype=np.int64)
        hold = max(int(params.hold_bars), 1)
        in_pos = False
        bars_held = 0
        for i in range(n):
            if in_pos:
                sig[i] = 1
                bars_held += 1
                if bars_held >= hold:
                    # Fixed-bar exit: position closes exactly hold bars after entry.
                    in_pos = False
                    bars_held = 0
            elif entry[i]:
                in_pos = True
                bars_held = 1
                sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
