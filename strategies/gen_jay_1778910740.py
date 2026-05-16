from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compress_window: int = 10
    atr_window: int = 20
    vol_window: int = 10
    entry_threshold: float = 0.55
    min_compress_ratio: float = 0.70


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778910740"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # atr_window bars for ATR, then compress_window more for the rolling-min
        # compression gate, plus 2 guard bars
        return max(params.atr_window, params.compress_window, params.vol_window) + params.compress_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        # True range and ATR
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Rolling range over compress_window
        roll_high = data["high"].rolling(params.compress_window, min_periods=params.compress_window).max()
        roll_low = data["low"].rolling(params.compress_window, min_periods=params.compress_window).min()
        roll_range = roll_high - roll_low

        # Compression ratio: actual range vs expected random-walk range (sqrt-N scaling)
        # ratio < 1 => tighter than random walk => elastic / compressed state
        expected_range = atr * (params.compress_window ** 0.5)
        compress_ratio = roll_range / expected_range.replace(0.0, np.nan)
        ind["compress_ratio"] = compress_ratio

        # Close position within rolling range: 0 = at low, 1 = at high
        safe_range = roll_range.replace(0.0, np.nan)
        range_pos = (data["close"] - roll_low) / safe_range
        ind["range_pos"] = range_pos.clip(0.0, 1.0)

        # Up-volume fraction over vol_window (bullish directional pressure)
        up_mask = (data["close"] >= data["open"]).astype(float)
        roll_up_vol = (data["volume"] * up_mask).rolling(params.vol_window, min_periods=params.vol_window).sum()
        roll_total_vol = data["volume"].rolling(params.vol_window, min_periods=params.vol_window).sum()
        vol_pressure = roll_up_vol / roll_total_vol.replace(0.0, np.nan)
        ind["vol_pressure"] = vol_pressure.clip(0.0, 1.0)

        # Volume-weighted Range Plasticity Score (VRPS)
        # High only when up-volume dominates AND price is near top of range
        ind["vrps"] = ind["vol_pressure"] * ind["range_pos"]

        # Recent compression flag: rolling-min of compress_ratio below threshold
        # within the last compress_window bars
        ind["was_compressed"] = (
            compress_ratio.rolling(params.compress_window, min_periods=1).min()
            < params.min_compress_ratio
        ).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["size"] = 1.0

        vrps = indicators["vrps"]
        was_compressed = indicators["was_compressed"]

        # Symmetric rule: long when VRPS exceeds threshold AND range was recently
        # compressed. The same condition drives exit when it flips (signal-reversal).
        active = (
            (vrps > params.entry_threshold) & (was_compressed > 0.5)
        ).astype(int)

        # Mandatory 1-bar shift: decision on bar N fills on bar N+1
        df["signal"] = active.shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
