from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    dd_window: int = 252
    trend_ma: int = 200
    trough_window: int = 20
    min_depth: float = 0.05
    hold_bars: int = 18
    size_base: float = 0.6
    size_depth_scale: float = 4.0
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778906710"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    @staticmethod
    def warmup_bars(params: GenA1Params) -> int:
        return int(params.dd_window + params.trough_window + 5)

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"].astype(float)

        dd_window = max(2, int(params.dd_window))
        trough_window = max(2, int(params.trough_window))
        trend_ma = max(2, int(params.trend_ma))

        roll_max = close.rolling(dd_window, min_periods=dd_window).max()
        dd = close / roll_max - 1.0
        trough = dd.rolling(trough_window, min_periods=trough_window).min()
        ma = close.rolling(trend_ma, min_periods=trend_ma).mean()

        dd_p1 = dd.shift(1)
        dd_p2 = dd.shift(2)
        recovering = (dd > dd_p1) & (dd_p1 > dd_p2)

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["trough"] = trough
        out["ma"] = ma
        out["recovering"] = recovering.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        close = data["close"].astype(float).to_numpy()
        dd = indicators["dd"].to_numpy(dtype=float)
        trough = indicators["trough"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        recovering = indicators["recovering"].fillna(0.0).to_numpy(dtype=float) > 0.5

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        if n == 0:
            df = pd.DataFrame(index=data.index)
            df["signal"] = signal
            df["size"] = size
            return SignalFrame(data=df, signal_column="signal", size_column="size")

        valid = np.isfinite(dd) & np.isfinite(trough) & np.isfinite(ma) & np.isfinite(close)
        min_depth = float(params.min_depth)
        depth_gate = valid & (trough <= -min_depth)
        regime_gate = valid & (close > ma)
        entry_ok = depth_gate & regime_gate & recovering

        hold_total = max(1, int(params.hold_bars))
        size_base = float(params.size_base)
        size_scale = float(params.size_depth_scale)
        size_max = float(params.size_max)

        hold = 0
        entry_size = 1.0
        i = 0
        while i < n:
            if hold > 0:
                signal[i] = 1
                size[i] = entry_size
                hold -= 1
                i += 1
                continue
            if entry_ok[i]:
                tension = max(0.0, (-float(trough[i])) - min_depth)
                entry_size = float(
                    np.clip(size_base + size_scale * tension, 0.5, max(0.5, size_max))
                )
                signal[i] = 1
                size[i] = entry_size
                hold = hold_total - 1
                i += 1
                continue
            i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float).clip(lower=0.1)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
