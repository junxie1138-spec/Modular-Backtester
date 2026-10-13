from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_span: int = 2
    medium_span: int = 5
    long_span: int = 9
    baseline_window: int = 16
    atr_window: int = 14
    entry_threshold: float = 1.0
    rearm_threshold: float = 0.25
    stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791317139"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, max(params.short_span, params.medium_span, params.long_span) + params.baseline_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.short_span, params.medium_span, params.long_span, params.baseline_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not (0 <= params.rearm_threshold < params.entry_threshold) or not np.isfinite(params.entry_threshold):
            raise ValueError("Require finite entry_threshold greater than rearm_threshold >= 0")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        close = data["close"].astype(float)
        previous = close.shift(1)
        ranges = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1)
        atr = ranges.max(axis=1).rolling(params.atr_window, min_periods=params.atr_window).mean().where(lambda x: x > 0)
        slopes = pd.concat([close.diff(span) / span for span in (params.short_span, params.medium_span, params.long_span)], axis=1)
        latent = slopes.median(axis=1, skipna=False)
        baseline = latent.shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).median()
        disagreement = slopes.max(axis=1, skipna=False) - slopes.min(axis=1, skipna=False)
        innovation = latent - baseline
        score = innovation / (disagreement + 0.1 * atr)
        consensus = pd.Series(np.select([(slopes > 0).all(axis=1), (slopes < 0).all(axis=1)], [1, -1], default=0), index=data.index)
        return pd.DataFrame({"ma200": close.rolling(200, min_periods=200).mean(), "atr": atr, "latent_slope": latent, "baseline_slope": baseline, "slope_disagreement": disagreement, "slope_innovation": innovation, "innovation_score": score, "consensus": consensus}, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        score = indicators["innovation_score"].to_numpy(dtype=float)
        consensus = indicators["consensus"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = np.nan
        stop = np.nan
        armed = True
        for i in range(len(data)):
            valid = np.isfinite([close[i], atr[i], ma[i], score[i], consensus[i]]).all() and atr[i] > 0
            if not valid:
                position = 0
                age = 0
                watermark = stop = np.nan
                continue
            if abs(score[i]) <= params.rearm_threshold:
                armed = True
            if position:
                age += 1
                if position == 1:
                    watermark = close[i] if not np.isfinite(watermark) else max(watermark, close[i])
                    candidate = watermark - params.stop_atr * atr[i]
                    stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                    stopped = close[i] <= stop
                    regime_broken = close[i] <= ma[i]
                else:
                    watermark = close[i] if not np.isfinite(watermark) else min(watermark, close[i])
                    candidate = watermark + params.stop_atr * atr[i]
                    stop = candidate if not np.isfinite(stop) else min(stop, candidate)
                    stopped = close[i] >= stop
                    regime_broken = close[i] >= ma[i]
                if stopped or regime_broken or age >= params.hold_bars:
                    position = 0
                    age = 0
                    watermark = stop = np.nan
                    continue
            elif armed:
                if close[i] > ma[i] and consensus[i] == 1 and score[i] >= params.entry_threshold:
                    position = 1
                elif close[i] < ma[i] and consensus[i] == -1 and score[i] <= -params.entry_threshold:
                    position = -1
                if position:
                    armed = False
                    age = 0
                    watermark = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100.0, 103.0, 105.0, 104.0, 102.0, 101.0, 100.0, 99.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0], "ma200": 90.0, "innovation_score": [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "consensus": 1}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params(stop_atr=2.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
