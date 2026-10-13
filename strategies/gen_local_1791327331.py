from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReleaseParams:
    frac_order: float = 0.45
    frac_window: int = 32
    channel_window: int = 20
    compression_window: int = 5
    compression_limit: float = 0.65
    breakout_z: float = 1.6
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ReleaseParams]):
    strategy_id = "gen_local_1791327331"

    @classmethod
    def params_type(cls):
        return ReleaseParams

    def warmup_bars(self, params: ReleaseParams) -> int:
        return int(max(
            params.frac_window + max(params.channel_window, params.compression_window) + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: ReleaseParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_window, params.channel_window, params.compression_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two")
        if params.hold_bars < 1 or params.stop_atr <= 0 or params.breakout_z <= 0:
            raise ValueError("Holding period, stop distance, and breakout threshold must be positive")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        if not (0 < params.compression_limit < 1):
            raise ValueError("compression_limit must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fd = close.copy()
        for k in range(1, params.frac_window):
            fd = fd + weights[k] * close.shift(k)
        out["fractional_close"] = fd
        out["reference_mean"] = fd.rolling(params.channel_window, min_periods=params.channel_window).mean().shift(1)
        out["reference_std"] = fd.rolling(params.channel_window, min_periods=params.channel_window).std(ddof=0).shift(1)
        out["short_std"] = fd.rolling(params.compression_window, min_periods=params.compression_window).std(ddof=0).shift(1)
        denominator = out["reference_std"].where(out["reference_std"] > 0)
        out["compression"] = out["short_std"] / denominator
        out["z"] = ((fd - out["reference_mean"]) / denominator).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        compressed = out["compression"] < params.compression_limit
        z = out["z"]
        long_entry = compressed & (z > params.breakout_z) & (z.shift(1) <= params.breakout_z)
        short_entry = compressed & (z < -params.breakout_z) & (z.shift(1) >= -params.breakout_z)
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        overshoot = (z.abs() / params.breakout_z - 1.0).clip(0.0, 1.0)
        out["entry_size"] = (params.min_size + (params.max_size - params.min_size) * overshoot).fillna(params.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReleaseParams) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        strengths = indicators["entry_size"].to_numpy(dtype=float)
        position = 0
        pending_fill = False
        entry_bar = -1
        entry_price = 0.0
        entry_atr = 0.0
        position_size = params.min_size
        for i in range(n):
            if pending_fill:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive")
                entry_price = opens[i]
                entry_bar = i
                pending_fill = False
            if position:
                stop_hit = np.isfinite(closes[i]) and position * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    position = 0
                else:
                    decisions[i] = position
                    sizes[i] = position_size
                continue
            if i + 1 < n and entries[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strengths[i]):
                position = int(entries[i])
                entry_atr = float(atr[i])
                position_size = float(np.clip(strengths[i], params.min_size, params.max_size))
                pending_fill = True
                decisions[i] = position
                sizes[i] = position_size
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 102.0, 103.0, 100.0, 100.0],
        "high": [101.0, 104.0, 104.0, 101.0, 101.0],
        "low": [99.0, 101.0, 98.0, 99.0, 99.0],
        "close": [100.0, 103.0, 99.0, 100.0, 100.0],
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 5,
        "entry_direction": [1, 0, 0, 0, 0],
        "entry_size": [0.55] * 5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, ReleaseParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(result.loc[index[1:3], "size"], 0.55)
