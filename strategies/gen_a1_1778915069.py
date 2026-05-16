from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticRecoveryParams:
    peak_window: int = 60
    roc_period: int = 10
    accel_smooth: int = 5
    dd_threshold: float = 0.04
    runup_threshold: float = 0.06
    profit_target: float = 0.06
    time_stop: int = 18
    tension_gain: float = 4.0
    max_size: float = 2.0


class GeneratedStrategy(BaseStrategy[ElasticRecoveryParams]):
    strategy_id = "gen_a1_1778915069"

    @classmethod
    def params_type(cls):
        return ElasticRecoveryParams

    @staticmethod
    def warmup_bars(params: ElasticRecoveryParams) -> int:
        roc_chain = int(params.roc_period) + int(params.accel_smooth) + 2
        return int(max(int(params.peak_window), roc_chain)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: ElasticRecoveryParams) -> pd.DataFrame:
        close = data["close"]
        peak_window = max(int(params.peak_window), 2)
        roc_period = max(int(params.roc_period), 1)
        accel_smooth = max(int(params.accel_smooth), 1)

        peak = close.rolling(peak_window, min_periods=2).max()
        trough = close.rolling(peak_window, min_periods=2).min()

        drawdown = close / peak.replace(0.0, np.nan) - 1.0
        runup = close / trough.replace(0.0, np.nan) - 1.0

        roc = close.pct_change(roc_period)
        accel = roc.diff(1).rolling(accel_smooth, min_periods=1).mean()

        out = pd.DataFrame(index=data.index)
        out["drawdown"] = drawdown
        out["runup"] = runup
        out["accel"] = accel
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticRecoveryParams) -> SignalFrame:
        idx = data.index
        n = len(data)

        close = data["close"].to_numpy(dtype=float)
        dd = np.nan_to_num(indicators["drawdown"].to_numpy(dtype=float), nan=0.0)
        ru = np.nan_to_num(indicators["runup"].to_numpy(dtype=float), nan=0.0)
        ac = np.nan_to_num(indicators["accel"].to_numpy(dtype=float), nan=0.0)

        dd_threshold = float(params.dd_threshold)
        runup_threshold = float(params.runup_threshold)
        profit_target = float(params.profit_target)
        time_stop = max(int(params.time_stop), 1)
        tension_gain = float(params.tension_gain)
        max_size = max(float(params.max_size), 1.0)

        long_entry = (dd <= -dd_threshold) & (ac > 0.0)
        short_entry = (ru >= runup_threshold) & (ac < 0.0)

        allow_short = bool(getattr(ctx, "allow_short", True))

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_size = 1.0

        for i in range(n):
            if position == 0:
                if long_entry[i]:
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                    tension = min(abs(dd[i]), 0.25)
                    entry_size = min(1.0 + tension_gain * tension, max_size)
                    signal[i] = 1
                    size[i] = entry_size
                elif allow_short and short_entry[i]:
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
                    tension = min(abs(ru[i]), 0.25)
                    entry_size = min(1.0 + tension_gain * tension, max_size)
                    signal[i] = -1
                    size[i] = entry_size
                else:
                    signal[i] = 0
                    size[i] = 1.0
            else:
                held = i - entry_idx
                if entry_price > 0.0:
                    pnl = (close[i] / entry_price - 1.0) * position
                else:
                    pnl = 0.0
                if pnl >= profit_target or held >= time_stop:
                    position = 0
                    signal[i] = 0
                    size[i] = 1.0
                else:
                    signal[i] = position
                    size[i] = entry_size

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = np.where(size > 0.0, size, 1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
