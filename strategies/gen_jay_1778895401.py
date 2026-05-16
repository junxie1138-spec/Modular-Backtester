from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    zscore_window: int = 20
    dd_window: int = 60
    dd_thresh: float = 0.03
    arm_level: float = -1.5
    fire_level: float = -0.5
    atr_window: int = 14
    trail_k: float = 2.0
    base_size: float = 0.95
    size_cap: float = 3.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778895401"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.ma_window, params.zscore_window, params.dd_window, params.atr_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(params.ma_window).mean()
        std = close.rolling(params.zscore_window).std()
        ind["zscore"] = (close - ma) / std.replace(0, np.nan)

        rolling_max = close.rolling(params.dd_window).max()
        ind["drawdown_frac"] = (close - rolling_max) / rolling_max.replace(0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        zscore = indicators["zscore"].to_numpy(dtype=float)
        dd_frac = indicators["drawdown_frac"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, params.base_size)

        armed = False
        arm_depth = 0.0
        entry_depth = 0.0
        in_trade = False
        hwm = 0.0

        denom = max(abs(params.arm_level), 1e-6)

        for i in range(n):
            z = zscore[i]
            dd = dd_frac[i]
            a = atr[i]
            c = close[i]

            if np.isnan(z) or np.isnan(dd) or np.isnan(a):
                continue

            if in_trade:
                if c > hwm:
                    hwm = c
                stop_price = hwm - params.trail_k * a
                if c < stop_price:
                    raw_signal[i] = 0
                    in_trade = False
                    entry_depth = 0.0
                else:
                    raw_signal[i] = 1
                    scale = float(np.clip(abs(entry_depth) / denom, 1.0, params.size_cap))
                    raw_size[i] = params.base_size * scale / params.size_cap
            else:
                in_drawdown = dd < -params.dd_thresh
                if in_drawdown:
                    if z < params.arm_level:
                        armed = True
                        if z < arm_depth:
                            arm_depth = z
                    if armed and z > params.fire_level:
                        entry_depth = arm_depth
                        scale = float(np.clip(abs(entry_depth) / denom, 1.0, params.size_cap))
                        raw_size[i] = params.base_size * scale / params.size_cap
                        raw_signal[i] = 1
                        in_trade = True
                        hwm = c
                        armed = False
                        arm_depth = 0.0
                else:
                    armed = False
                    arm_depth = 0.0

        df = pd.DataFrame(
            {"signal": raw_signal, "size": raw_size},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
