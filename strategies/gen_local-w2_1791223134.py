from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalTensionParams:
    fractional_order: float = 0.45
    fractional_window: int = 24
    percentile_window: int = 126
    entry_percentile: float = 0.90
    breakout_window: int = 10
    atr_window: int = 14
    trail_atr_mult: float = 1.8
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[FractionalTensionParams]):
    strategy_id = "gen_local-w2_1791223134"

    @classmethod
    def params_type(cls) -> type[FractionalTensionParams]:
        return FractionalTensionParams

    def warmup_bars(self, params: FractionalTensionParams) -> int:
        return int(max(
            params.fractional_window + params.percentile_window,
            params.breakout_window + 1,
            params.atr_window + 1,
        ))

    def indicators(
        self,
        data: pd.DataFrame,
        params: FractionalTensionParams,
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(
            params.atr_window,
            min_periods=params.atr_window,
        ).mean()

        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (
                params.fractional_order - lag + 1
            ) / lag

        fractional_close = pd.Series(0.0, index=data.index, dtype=float)
        for lag, weight in enumerate(weights):
            fractional_close = fractional_close + weight * close.shift(lag)

        tension = fractional_close / atr.replace(0.0, np.nan)
        history = tension.rolling(
            params.percentile_window,
            min_periods=params.percentile_window,
        )

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["fractional_close"] = fractional_close
        out["tension"] = tension
        out["upper_tension"] = history.quantile(params.entry_percentile).shift(1)
        out["lower_tension"] = history.quantile(1.0 - params.entry_percentile).shift(1)
        out["prior_high"] = close.rolling(
            params.breakout_window,
            min_periods=params.breakout_window,
        ).max().shift(1)
        out["prior_low"] = close.rolling(
            params.breakout_window,
            min_periods=params.breakout_window,
        ).min().shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalTensionParams,
    ) -> SignalFrame:
        close_series = data["close"].astype(float)
        valid = indicators[
            ["atr", "tension", "upper_tension", "lower_tension", "prior_high", "prior_low"]
        ].notna().all(axis=1)
        long_entry = (
            valid
            & indicators["tension"].gt(indicators["upper_tension"])
            & close_series.gt(indicators["prior_high"])
        ).to_numpy(dtype=bool)
        short_entry = (
            valid
            & indicators["tension"].lt(indicators["lower_tension"])
            & close_series.lt(indicators["prior_low"])
        ).to_numpy(dtype=bool)

        close = close_series.to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        high_water = 0.0
        low_water = 0.0
        stop = 0.0
        first_decision = self.warmup_bars(params) - 1

        for i in range(len(data)):
            price = close[i]
            current_atr = atr[i]
            bar_valid = np.isfinite(price) and np.isfinite(current_atr) and current_atr > 0.0

            if position == 0:
                if i < first_decision or not bar_valid:
                    continue
                if long_entry[i]:
                    position = 1
                    held = 0
                    high_water = price
                    stop = price - params.trail_atr_mult * current_atr
                elif short_entry[i]:
                    position = -1
                    held = 0
                    low_water = price
                    stop = price + params.trail_atr_mult * current_atr
                signal[i] = position
                continue

            held += 1
            if not bar_valid:
                position = 0
                continue

            if position == 1:
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.trail_atr_mult * current_atr)
                exit_now = price <= stop or held >= params.max_hold
            else:
                low_water = min(low_water, price)
                stop = min(stop, low_water + params.trail_atr_mult * current_atr)
                exit_now = price >= stop or held >= params.max_hold

            if exit_now:
                position = 0
            else:
                signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
