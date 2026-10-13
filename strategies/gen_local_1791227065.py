from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_span: int = 8
    surprise_span: int = 21
    pressure_window: int = 5
    atr_window: int = 14
    innovation_atr: float = 0.04
    pressure_floor: float = 0.12
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791227065"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.trend_span + params.surprise_span + 1, params.pressure_window, params.atr_window)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = true_range.ewm(span=params.atr_window, adjust=False, min_periods=params.atr_window).mean().fillna(0.0)

        log_close = np.log(close.where(close > 0))
        slope = log_close.ewm(span=params.trend_span, adjust=False, min_periods=params.trend_span).mean().diff()
        expected_slope = slope.ewm(span=params.surprise_span, adjust=False, min_periods=params.surprise_span).mean().shift(1)
        out["innovation"] = (slope - expected_slope).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        bar_range = (data["high"] - data["low"]).replace(0, np.nan)
        close_location = ((2 * close - data["high"] - data["low"]) / bar_range).fillna(0.0)
        volume = data["volume"].clip(lower=0)
        weighted_location = (close_location * volume).rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        rolling_volume = volume.rolling(params.pressure_window, min_periods=params.pressure_window).sum().replace(0, np.nan)
        out["pressure"] = (weighted_location / rolling_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        innovation = indicators["innovation"].to_numpy(dtype=float)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = False
        high_water = -np.inf
        stop = -np.inf
        held = 0

        for i, close in enumerate(closes):
            if position:
                held += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close <= stop or held >= params.max_hold:
                    position = False
                else:
                    signal[i] = 1
            elif (
                i >= self.warmup_bars(params)
                and np.isfinite(close)
                and atr[i] > 0
                and innovation[i] > params.innovation_atr * atr[i] / close
                and pressure[i] > params.pressure_floor
            ):
                position = True
                high_water = -np.inf
                stop = -np.inf
                held = 0
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
