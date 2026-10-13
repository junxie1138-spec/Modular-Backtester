from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 40
    recent_window: int = 3
    baseline_window: int = 12
    atr_window: int = 14
    min_depth_atr: float = 2.0
    change_threshold: float = 0.15
    pressure_threshold: float = 0.12
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790987654"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.peak_window + params.recent_window + params.baseline_window,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            (data["high"] - data["low"]).abs(),
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        out["peak"] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max().replace(0.0, np.nan)
        out["drawdown"] = 1.0 - close / out["peak"]
        out["drawdown_change"] = out["drawdown"].diff()
        out["recent_change"] = out["drawdown_change"].rolling(
            params.recent_window, min_periods=params.recent_window
        ).mean()
        out["baseline_change"] = out["drawdown_change"].shift(
            params.recent_window
        ).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["atr_fraction"] = out["atr"] / close.replace(0.0, np.nan)
        out["depth_atr"] = out["drawdown"] / out["atr_fraction"]
        out["depth_break"] = (
            out["recent_change"] - out["baseline_change"]
        ) / out["atr_fraction"]
        out["candle_location"] = (
            (2.0 * close - data["high"] - data["low"])
            / (data["high"] - data["low"]).replace(0.0, np.nan)
        ).fillna(0.0).clip(-1.0, 1.0)
        out["signed_volume"] = (
            out["candle_location"] * data["volume"].clip(lower=0.0)
        )
        out["recent_signed_volume"] = out["signed_volume"].rolling(
            params.recent_window, min_periods=params.recent_window
        ).sum()
        out["recent_volume"] = data["volume"].clip(lower=0.0).rolling(
            params.recent_window, min_periods=params.recent_window
        ).sum()
        out["baseline_signed_volume"] = out["signed_volume"].shift(
            params.recent_window
        ).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).sum()
        out["baseline_volume"] = data["volume"].clip(lower=0.0).shift(
            params.recent_window
        ).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).sum()
        out["recent_pressure"] = (
            out["recent_signed_volume"]
            / out["recent_volume"].replace(0.0, np.nan)
        )
        out["baseline_pressure"] = (
            out["baseline_signed_volume"]
            / out["baseline_volume"].replace(0.0, np.nan)
        )
        out["pressure_break"] = (
            out["recent_pressure"] - out["baseline_pressure"]
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        deep = indicators["depth_atr"].ge(params.min_depth_atr)
        long_agreement = (
            deep
            & indicators["depth_break"].lt(-params.change_threshold)
            & indicators["pressure_break"].gt(params.pressure_threshold)
            & indicators["recent_pressure"].gt(0.0)
        )
        short_agreement = (
            deep
            & indicators["depth_break"].gt(params.change_threshold)
            & indicators["pressure_break"].lt(-params.pressure_threshold)
            & indicators["recent_pressure"].lt(0.0)
        )
        long_event = long_agreement & ~long_agreement.shift(1, fill_value=False)
        short_event = short_agreement & ~short_agreement.shift(1, fill_value=False)
        candidates = np.select([long_event, short_event], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, low_water = np.nan, np.nan
        high_stop, short_stop = -np.inf, np.inf

        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0.0:
                position, age = 0, 0
                signals[i] = 0
                continue

            if position != 0:
                age += 1
                # Tracking starts on the fill bar, after the entry decision.
                high_water = price if not np.isfinite(high_water) else max(high_water, price)
                low_water = price if not np.isfinite(low_water) else min(low_water, price)
                # The required highest-close stop can only ratchet upward.
                high_stop = max(high_stop, high_water - params.trail_atr * atr)
                # Shorts additionally have a downward-ratcheting adverse-move stop.
                short_stop = min(short_stop, low_water + params.trail_atr * atr)
                exit_now = price <= high_stop or age >= params.max_hold
                if position == -1:
                    exit_now = exit_now or price >= short_stop
                if exit_now:
                    position, age = 0, 0
                signals[i] = position
                continue

            if candidates[i] != 0:
                position = int(candidates[i])
                age = 0
                high_water, low_water = np.nan, np.nan
                high_stop, short_stop = -np.inf, np.inf
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
