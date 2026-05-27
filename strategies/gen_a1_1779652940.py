from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CLVExhaustionParams:
    clv_window: int = 5
    range_short: int = 5
    range_long: int = 30
    clv_low_threshold: float = 0.30
    range_contraction_threshold: float = 0.85
    trend_filter_ma: int = 200
    hold_bars: int = 2
    base_size: float = 0.5
    max_size_multiplier: float = 2.0
    use_trend_filter: bool = True


class GeneratedStrategy(BaseStrategy[CLVExhaustionParams]):
    strategy_id = "gen_a1_1779652940"

    @classmethod
    def params_type(cls):
        return CLVExhaustionParams

    @classmethod
    def warmup_bars(cls, params: CLVExhaustionParams) -> int:
        return int(max(params.trend_filter_ma, params.range_long) + params.clv_window + 2)

    def indicators(self, data: pd.DataFrame, params: CLVExhaustionParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = (high - low)
        # avoid div-by-zero on flat (doji) bars; NaN propagates and is handled downstream
        rng_safe = rng.where(rng > 0.0, np.nan)

        clv = (close - low) / rng_safe
        clv = clv.clip(lower=0.0, upper=1.0)
        clv_mean = clv.rolling(params.clv_window, min_periods=params.clv_window).mean()

        range_short = rng.rolling(params.range_short, min_periods=params.range_short).mean()
        range_long = rng.rolling(params.range_long, min_periods=params.range_long).mean()
        range_long_safe = range_long.where(range_long > 0.0, np.nan)
        range_ratio = range_short / range_long_safe

        ma_trend = close.rolling(params.trend_filter_ma, min_periods=params.trend_filter_ma).mean()

        clv_thr = max(float(params.clv_low_threshold), 1e-6)
        rng_thr = max(float(params.range_contraction_threshold), 1e-6)
        clv_pressure = ((clv_thr - clv_mean) / clv_thr).clip(lower=0.0)
        range_pressure = ((rng_thr - range_ratio) / rng_thr).clip(lower=0.0)
        exhaustion_score = (clv_pressure * range_pressure).clip(lower=0.0, upper=1.0)

        out = pd.DataFrame(index=data.index)
        out["clv_mean"] = clv_mean
        out["range_ratio"] = range_ratio
        out["ma_trend"] = ma_trend
        out["exhaustion_score"] = exhaustion_score
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CLVExhaustionParams,
    ) -> SignalFrame:
        n = len(data)
        clv_mean = indicators["clv_mean"].to_numpy(dtype=np.float64)
        range_ratio = indicators["range_ratio"].to_numpy(dtype=np.float64)
        ma_trend = indicators["ma_trend"].to_numpy(dtype=np.float64)
        score = indicators["exhaustion_score"].to_numpy(dtype=np.float64)
        close = data["close"].to_numpy(dtype=np.float64)

        signal = np.zeros(n, dtype=np.int64)
        size = np.full(n, float(params.base_size), dtype=np.float64)

        hold_bars = max(int(params.hold_bars), 1)
        bars_remaining = 0
        current_size = float(params.base_size)
        clv_thr = float(params.clv_low_threshold)
        rng_thr = float(params.range_contraction_threshold)
        use_trend = bool(params.use_trend_filter)
        max_mult = float(params.max_size_multiplier)
        base_size = float(params.base_size)

        for i in range(n):
            if bars_remaining > 0:
                signal[i] = 1
                size[i] = current_size
                bars_remaining -= 1
                continue

            if (
                np.isnan(clv_mean[i])
                or np.isnan(range_ratio[i])
                or np.isnan(score[i])
                or (use_trend and np.isnan(ma_trend[i]))
            ):
                continue

            trend_ok = (not use_trend) or (close[i] > ma_trend[i])
            if (
                clv_mean[i] < clv_thr
                and range_ratio[i] < rng_thr
                and trend_ok
            ):
                mult = 1.0 + min(max(score[i], 0.0), 1.0) * (max_mult - 1.0)
                current_size = base_size * mult
                if current_size <= 0.0:
                    current_size = base_size
                signal[i] = 1
                size[i] = current_size
                bars_remaining = hold_bars - 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        size_shifted = pd.Series(size, index=data.index).shift(1).fillna(base_size).astype(float)
        # guarantee strictly positive size column
        size_shifted = size_shifted.where(size_shifted > 0.0, base_size)
        df["size"] = size_shifted
        return SignalFrame(data=df, signal_column="signal", size_column="size")
