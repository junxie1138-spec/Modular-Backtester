from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRecoveryParams:
    peak_window: int = 60
    arm_depth: float = 0.05
    fire_depth: float = 0.02
    cap_depth: float = 0.15
    min_size: float = 0.4
    profit_target: float = 0.04
    max_hold: int = 10
    trend_window: int = 200
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[DrawdownRecoveryParams]):
    strategy_id = "gen_a1_1778891783"

    @classmethod
    def params_type(cls):
        return DrawdownRecoveryParams

    def warmup_bars(self, params: DrawdownRecoveryParams) -> int:
        return int(max(params.peak_window, params.trend_window) + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownRecoveryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = close / peak - 1.0
        trend_ma = close.rolling(params.trend_window, min_periods=params.trend_window).mean()
        out = pd.DataFrame(index=data.index)
        out["peak"] = peak
        out["drawdown"] = drawdown
        out["trend_ma"] = trend_ma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownRecoveryParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        trend_ma = indicators["trend_ma"].to_numpy(dtype=float)
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.zeros(n, dtype=float)

        arm_depth = float(params.arm_depth)
        fire_depth = float(params.fire_depth)
        span = max(float(params.cap_depth) - arm_depth, 1e-6)
        min_size = float(params.min_size)
        profit_target = float(params.profit_target)
        max_hold = int(params.max_hold)

        armed = False
        trough = 0.0
        in_pos = False
        bars_held = 0
        entry_price = 0.0
        entry_size = 0.0

        for i in range(n):
            d = dd[i]
            c = close[i]
            if not np.isfinite(d) or not np.isfinite(c) or c <= 0.0:
                continue

            exited = False
            if in_pos:
                bars_held += 1
                gain = c / entry_price - 1.0
                if gain >= profit_target or bars_held >= max_hold:
                    in_pos = False
                    exited = True
                else:
                    raw_signal[i] = 1
                    raw_size[i] = entry_size

            fired = False
            if not armed:
                if d <= -arm_depth:
                    armed = True
                    trough = d
            else:
                if d < trough:
                    trough = d
                if d >= -fire_depth:
                    fired = True
                    armed = False

            if fired and not in_pos and not exited:
                regime_ok = True
                if params.use_trend_filter:
                    m = trend_ma[i]
                    regime_ok = bool(np.isfinite(m)) and c > m
                if regime_ok:
                    depth = abs(trough)
                    ratio = (depth - arm_depth) / span
                    if ratio < 0.0:
                        ratio = 0.0
                    elif ratio > 1.0:
                        ratio = 1.0
                    size = min_size + (1.0 - min_size) * ratio
                    in_pos = True
                    bars_held = 0
                    entry_price = c
                    entry_size = size
                    raw_signal[i] = 1
                    raw_size[i] = size

            if fired:
                trough = 0.0

        signal = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        size = pd.Series(raw_size, index=data.index).shift(1)
        size = size.where(size > 0.0, 1.0).fillna(1.0).astype(float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
