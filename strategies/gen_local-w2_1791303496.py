from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_bars: int = 63
    progress_bars: int = 5
    atr_bars: int = 14
    depletion_threshold: float = 0.35
    persistence: float = 0.8
    minimum_depth: float = 0.01
    trailing_atr: float = 3.0
    holding_bars: int = 20


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791303496"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(200, params.peak_bars + params.progress_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if params.peak_bars < 2 or params.progress_bars < 1 or params.atr_bars < 1:
            raise ValueError("Indicator lookbacks must be positive; peak_bars must be at least two.")
        if params.holding_bars < 1 or params.trailing_atr <= 0:
            raise ValueError("Holding duration and trailing multiplier must be positive.")
        if not 0 < params.persistence <= 1 or params.minimum_depth <= 0 or params.depletion_threshold <= 0:
            raise ValueError("Depth thresholds must be positive and persistence must be in (0, 1].")
        close = data["close"]
        peak = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = (1.0 - close.div(peak.where(peak > 0))).clip(lower=0)
        previous_depth = depth.shift(params.progress_bars)
        depth_change = depth.diff()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        result = pd.DataFrame(index=data.index)
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        result["peak"] = peak
        result["previous_peak"] = peak.shift(params.progress_bars)
        result["depth"] = depth
        result["previous_depth"] = previous_depth
        result["depth_change"] = depth_change
        result["depletion"] = (previous_depth - depth).div(previous_depth.where(previous_depth > 0))
        result["erosion_persistence"] = (depth_change < 0).astype(float).where(depth_change.notna()).rolling(params.progress_bars, min_periods=params.progress_bars).mean()
        result["expansion_persistence"] = (depth_change > 0).astype(float).where(depth_change.notna()).rolling(params.progress_bars, min_periods=params.progress_bars).mean()
        result["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0)
        stable_peak = indicators["peak"].eq(indicators["previous_peak"])
        eligible = ready & stable_peak & (indicators["previous_depth"] >= params.minimum_depth)
        long_entry = eligible & (data["close"] > indicators["ma200"]) & (indicators["depletion"] >= params.depletion_threshold) & (indicators["erosion_persistence"] >= params.persistence)
        short_entry = eligible & (data["close"] < indicators["ma200"]) & (indicators["depletion"] <= -params.depletion_threshold) & (indicators["expansion_persistence"] >= params.persistence)
        entries = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        valid = ready.to_numpy(dtype=bool) & np.isfinite(close)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            if not valid[i]:
                position = 0
                continue
            if position:
                stopped = close[i] <= stop if position == 1 else close[i] >= stop
                expired = i - entry_bar >= params.holding_bars
                if stopped or expired:
                    position = 0
                    continue
                if position == 1:
                    extreme = max(extreme, close[i])
                    stop = max(stop, extreme - params.trailing_atr * atr[i])
                else:
                    extreme = min(extreme, close[i])
                    stop = min(stop, extreme + params.trailing_atr * atr[i])
                stopped = close[i] <= stop if position == 1 else close[i] >= stop
                if stopped:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i
                extreme = close[i]
                stop = extreme - position * params.trailing_atr * atr[i]
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    for direction in (1, -1):
        closes = np.array([100, 101, 104, 105, 99, 99, 99], dtype=float)
        if direction == -1:
            closes = 200 - closes
        data = pd.DataFrame({"close": closes}, index=index)
        indicators = pd.DataFrame({
            "ma200": 90.0 if direction == 1 else 110.0,
            "atr": 1.0, "peak": 120.0, "previous_peak": 120.0,
            "previous_depth": 0.02,
            "depletion": [direction * 0.5, 0, 0, 0, 0, 0, 0],
            "erosion_persistence": 1.0,
            "expansion_persistence": 1.0,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0]
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
