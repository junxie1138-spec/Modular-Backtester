from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownReleaseParams:
    peak_window: int = 40
    burden_window: int = 8
    recovery_bars: int = 3
    trend_window: int = 60
    minimum_depth: float = 0.015
    release_threshold: float = 0.35
    hold_bars: int = 4
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownReleaseParams]):
    strategy_id = "gen_local-w2_1791209585"

    @classmethod
    def params_type(cls) -> type[DrawdownReleaseParams]:
        return DrawdownReleaseParams

    def warmup_bars(self, params: DrawdownReleaseParams) -> int:
        peak = max(2, int(params.peak_window))
        burden = max(2, int(params.burden_window))
        recovery = max(1, int(params.recovery_bars))
        trend = max(1, int(params.trend_window))
        return int(max(peak + burden - 1, peak + recovery, trend + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownReleaseParams) -> pd.DataFrame:
        peak_window = max(2, int(params.peak_window))
        burden_window = max(2, int(params.burden_window))
        recovery_bars = max(1, int(params.recovery_bars))
        trend_window = max(1, int(params.trend_window))
        close = data["close"]
        peak = data["high"].rolling(peak_window, min_periods=peak_window).max()
        depth = (1.0 - close / peak.where(peak > 0)).clip(lower=0.0)
        energy = depth.pow(2)
        burden = energy.rolling(burden_window, min_periods=burden_window).mean()
        prior_depth = depth.shift(recovery_bars)
        prior_peak = peak.shift(recovery_bars)
        release = (energy.shift(recovery_bars) - energy) / burden.where(burden > 0)
        trend = close / close.shift(trend_window).where(close.shift(trend_window) > 0) - 1.0
        last_gain = close.diff()
        ready = release.notna() & trend.notna() & prior_depth.notna() & last_gain.notna()
        stable_peak = peak.eq(prior_peak)
        eligible = (
            ready
            & stable_peak
            & (prior_depth >= max(0.0, params.minimum_depth))
            & (depth > 0.0)
            & (release >= max(0.000001, params.release_threshold))
            & (trend > 0.0)
            & (last_gain > 0.0)
        )
        confidence = (release / (2.0 * max(0.000001, params.release_threshold))).clip(0.0, 1.0)
        result = pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "energy": energy,
            "burden": burden,
            "prior_depth": prior_depth,
            "prior_peak": prior_peak,
            "release": release,
            "trend": trend,
            "last_gain": last_gain,
            "ready": ready,
            "stable_peak": stable_peak,
            "eligible": eligible,
            "confidence": confidence,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownReleaseParams) -> SignalFrame:
        hold_bars = max(1, int(params.hold_bars))
        minimum_size = float(np.clip(params.minimum_size, 0.000001, 1.0))
        maximum_size = float(np.clip(params.maximum_size, minimum_size, 1.0))
        eligible = indicators["eligible"].to_numpy(dtype=bool)
        confidence = indicators["confidence"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), minimum_size, dtype=float)
        entry_bar = -1
        entry_size = minimum_size
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar >= hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
                    size[i] = entry_size
            elif eligible[i]:
                entry_bar = i
                entry_size = minimum_size + (maximum_size - minimum_size) * confidence[i]
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
