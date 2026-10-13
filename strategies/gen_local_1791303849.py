from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    frac_order: float = 0.45
    frac_window: int = 32
    peak_window: int = 63
    percentile_window: int = 126
    depth_percentile: float = 0.70
    tail_percentile: float = 0.20
    memory_bars: int = 5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791303849"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.frac_window, params.peak_window) + params.percentile_window + params.memory_bars + 2)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not (0 < params.frac_order < 1 and 0 < params.tail_percentile < 0.5 and 0 < params.depth_percentile < 1):
            raise ValueError("Invalid fractional order or percentile")
        if min(params.frac_window, params.peak_window, params.percentile_window, params.memory_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        # ponytail: finite fractional memory; enlarge frac_window if truncation sensitivity matters.
        fd = close * weights[0]
        for k in range(1, params.frac_window):
            fd = fd + close.shift(k) * weights[k]
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fd
        out["fractional_memory"] = fd / close
        history = out["fractional_memory"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        out["fd_low"] = history.quantile(params.tail_percentile)
        out["fd_mid"] = history.quantile(0.5)
        out["fd_high"] = history.quantile(1 - params.tail_percentile)
        out["drawdown"] = 1 - close / close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth_threshold"] = out["drawdown"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.depth_percentile)
        out["depth_change"] = out["drawdown"].diff()
        ready = out[["fractional_memory", "fd_low", "fd_mid", "fd_high", "drawdown", "depth_threshold", "depth_change"]].notna().all(axis=1)
        out["lower_tail"] = ready & (out["fractional_memory"] < out["fd_low"])
        out["upper_tail"] = ready & (out["fractional_memory"] > out["fd_high"])
        out["recent_lower_tail"] = out["lower_tail"].shift(1).rolling(params.memory_bars, min_periods=params.memory_bars).max().fillna(0).astype(bool)
        out["recent_upper_tail"] = out["upper_tail"].shift(1).rolling(params.memory_bars, min_periods=params.memory_bars).max().fillna(0).astype(bool)
        out["median_distance"] = out["fractional_memory"] - out["fd_mid"]
        deep = ready & (out["drawdown"] > 0) & (out["drawdown"] >= out["depth_threshold"])
        out["long_entry"] = (deep & out["recent_lower_tail"] & (out["median_distance"] > 0) & (out["median_distance"].shift(1) <= 0) & (out["depth_change"] < 0)).fillna(False)
        out["short_entry"] = (deep & out["recent_upper_tail"] & (out["median_distance"] < 0) & (out["median_distance"].shift(1) >= 0) & (out["depth_change"] > 0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        for i in range(len(data)):
            if position != 0 and i - entry_bar >= params.hold_bars:
                position = 0
            elif position == 0:
                if longs[i]:
                    position, entry_bar = 1, i
                elif shorts[i]:
                    position, entry_bar = -1, i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = RecoveryParams()
    d = pd.DataFrame(index=pd.date_range("2020-01-01", periods=30))
    ind = pd.DataFrame({"long_entry": False, "short_entry": False}, index=d.index)
    ind.iloc[0:12, 0] = True
    ind.iloc[12:30, 1] = True
    result = GeneratedStrategy().generate_signals(d, ind, None, p).data
    expected = np.zeros(30, dtype=int)
    expected[1:9] = 1
    expected[10:18] = 1
    expected[19:27] = -1
    expected[28:30] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
