from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    percentile_window: int = 60
    compression_percentile: float = 0.25
    streak_bars: int = 3
    atr_bars: int = 14
    trail_atr: float = 0.9


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790894275"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.percentile_window + 1, params.streak_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["relative_range"] = (data["high"] - data["low"]) / close.where(close > 0)
        out["compression_threshold"] = out["relative_range"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.compression_percentile).shift(1)
        compressed = (
            out["relative_range"].le(out["compression_threshold"])
            & out["compression_threshold"].notna()
            & out["relative_range"].notna()
        )
        out["compressed"] = compressed
        groups = (~compressed).cumsum()
        out["compression_streak"] = compressed.astype(int).groupby(groups).cumsum().clip(upper=params.streak_bars)
        out["prior_streak"] = out["compression_streak"].shift(1).fillna(0)
        out["envelope_high"] = data["high"].rolling(
            params.streak_bars, min_periods=params.streak_bars
        ).max().shift(1)
        out["envelope_low"] = data["low"].rolling(
            params.streak_bars, min_periods=params.streak_bars
        ).min().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        inside = close.ge(out["envelope_low"]) & close.le(out["envelope_high"])
        ready = (
            out["prior_streak"].ge(params.streak_bars)
            & ~compressed
            & inside
            & out["atr"].gt(0)
            & out["compression_threshold"].notna()
        )
        upper_escape = data["high"].gt(out["envelope_high"])
        lower_escape = data["low"].lt(out["envelope_low"])
        out["entry"] = np.select([
            ready & lower_escape & ~upper_escape,
            ready & upper_escape & ~lower_escape,
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    watermark = max(watermark, close)
                    stop = max(stop, watermark - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    watermark = min(watermark, close)
                    stop = min(stop, watermark + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or age >= 2:
                    position = 0
                    age = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                watermark = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
