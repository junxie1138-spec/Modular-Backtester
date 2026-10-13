from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVisibilityParams:
    lookback: int = 12
    triangle_share: float = 0.65
    stop_atr: float = 1.5
    atr_period: int = 14
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[SeasonalVisibilityParams]):
    strategy_id = "gen_local_1790991075"

    @classmethod
    def params_type(cls):
        return SeasonalVisibilityParams

    def warmup_bars(self, params: SeasonalVisibilityParams) -> int:
        return int(max(200, 7 * (max(3, params.lookback) + 2) + 2,
                       max(1, params.atr_period) + 1))

    def indicators(self, data: pd.DataFrame,
                   params: SeasonalVisibilityParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        result["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        atr_period = max(1, int(params.atr_period))
        result["atr"] = result["true_range"].rolling(
            atr_period, min_periods=atr_period).mean()
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        result["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        result["origin_weekday"] = result["weekday"].shift(2)
        # At today's close this outcome is fully mature: entry was the
        # open after the decision two bars ago, followed by two sessions.
        denominator = data["open"].shift(1).replace(0.0, np.nan)
        result["mature_return"] = (close / denominator - 1.0).replace(
            [np.inf, -np.inf], np.nan)
        lookback = max(3, int(params.lookback))
        result["seasonal_mean"] = np.nan
        result["triangle_fraction"] = np.nan
        result["triangle_count"] = np.nan
        for weekday in range(5):
            outcomes = result.loc[
                result["origin_weekday"].eq(weekday), "mature_return"]
            left = outcomes.shift(2)
            middle = outcomes.shift(1)
            valid = outcomes.notna() & middle.notna() & left.notna()
            # Consecutive vertices always share edges. The outer edge
            # closes a horizontal-visibility triangle iff the middle
            # observation lies strictly below both endpoints.
            triangle = (middle < pd.concat([left, outcomes], axis=1).min(axis=1))
            profitable_valley = triangle & left.gt(0.0) & outcomes.gt(0.0) & middle.lt(0.0)
            triangle_values = triangle.astype(float).where(valid)
            valley_values = profitable_valley.astype(float).where(valid)
            count = triangle_values.rolling(lookback, min_periods=lookback).sum()
            valley_count = valley_values.rolling(lookback, min_periods=lookback).sum()
            fraction = valley_count / count.replace(0.0, np.nan)
            mean = outcomes.rolling(lookback, min_periods=lookback).mean()
            result[f"weekday_{weekday}_mean"] = mean.reindex(data.index).ffill()
            result[f"weekday_{weekday}_triangles"] = count.reindex(data.index).ffill()
            result[f"weekday_{weekday}_fraction"] = fraction.reindex(data.index).ffill()
            mask = result["weekday"].eq(weekday)
            result.loc[mask, "seasonal_mean"] = result.loc[mask, f"weekday_{weekday}_mean"]
            result.loc[mask, "triangle_count"] = result.loc[mask, f"weekday_{weekday}_triangles"]
            result.loc[mask, "triangle_fraction"] = result.loc[mask, f"weekday_{weekday}_fraction"]
        return result

    def generate_signals(self, data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: SeasonalVisibilityParams) -> SignalFrame:
        regime = data["close"].gt(indicators["ma200"]).fillna(False)
        eligible = (
            regime
            & indicators["seasonal_mean"].gt(0.0)
            & indicators["triangle_count"].ge(2.0)
            & indicators["triangle_fraction"].ge(params.triangle_share)
            & indicators["atr"].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        above_ma = regime.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        frozen_atr = np.nan
        stop = np.nan
        hold_days = min(2, max(1, int(params.hold_days)))
        stop_multiple = max(0.01, float(params.stop_atr))
        for i in range(len(data)):
            if active:
                if held == 0:
                    # The prior close's decision fills at this bar's open.
                    # Freeze both the entry price and decision-time ATR.
                    if not np.isfinite(opening[i]) or opening[i] <= 0.0:
                        active = False
                        continue
                    stop = opening[i] - stop_multiple * frozen_atr
                held += 1
                if (not np.isfinite(close[i]) or close[i] < stop
                        or held >= hold_days or not above_ma[i]):
                    active = False
                    held = 0
                    continue
                signals[i] = 1
            elif eligible[i] and np.isfinite(atr[i]):
                active = True
                held = 0
                frozen_atr = atr[i]
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
