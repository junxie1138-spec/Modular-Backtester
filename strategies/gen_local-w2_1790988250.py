from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    atr_window: int = 5
    baseline_window: int = 60
    memory_bars: int = 504
    holding_bars: int = 18
    match_radius: float = 0.45
    minimum_analogs: int = 4
    forecast_threshold: float = 0.008
    minimum_win_rate: float = 0.60
    trail_atr: float = 2.8


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1790988250"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.memory_bars + params.baseline_window + 2 * params.atr_window + params.holding_bars + 1)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if (params.atr_window < 1 or params.baseline_window < params.atr_window
                or params.holding_bars < 1 or params.minimum_analogs < 1
                or params.match_radius <= 0 or params.trail_atr <= 0
                or not 0 <= params.minimum_win_rate <= 1):
            raise ValueError("Invalid strategy parameters")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["atr_fraction"] = out["atr"] / close.where(close > 0)
        out["volatility_baseline"] = out["atr_fraction"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        ratio = out["atr_fraction"] / out["volatility_baseline"].where(out["volatility_baseline"] > 0)
        out["shape_now"] = np.log(ratio.where(ratio > 0))
        out["shape_middle"] = out["shape_now"].shift(params.atr_window)
        out["shape_early"] = out["shape_now"].shift(2 * params.atr_window)
        out["log_volatility"] = np.log(out["atr_fraction"].where(out["atr_fraction"] > 0))
        out["matured_return"] = np.log(
            close.where(close > 0) / close.shift(params.holding_bars).where(close.shift(params.holding_bars) > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)

        weight_sum = pd.Series(0.0, index=data.index)
        return_sum = pd.Series(0.0, index=data.index)
        win_sum = pd.Series(0.0, index=data.index)
        analog_count = pd.Series(0, index=data.index, dtype=int)
        first_lag = params.baseline_window + 3 * params.atr_window + params.holding_bars
        features = ["shape_now", "shape_middle", "shape_early"]
        for lag in range(first_lag, params.memory_bars + 1, params.holding_bars):
            distance_squared = pd.Series(0.0, index=data.index)
            for feature in features:
                distance_squared += (out[feature] - out[feature].shift(lag)) ** 2 / 3.0
            distance_squared += 0.25 * (out["log_volatility"] - out["log_volatility"].shift(lag)) ** 2
            # The historical endpoint's outcome has already fully matured.
            outcome = out["matured_return"].shift(lag - params.holding_bars)
            valid = distance_squared.le(params.match_radius ** 2) & outcome.notna()
            weight = np.exp(-2.0 * distance_squared / params.match_radius ** 2).where(valid, 0.0).fillna(0.0)
            weight_sum += weight
            return_sum += weight * outcome.fillna(0.0)
            win_sum += weight * outcome.gt(0).astype(float)
            analog_count += valid.astype(int)

        out["analog_count"] = analog_count
        out["analog_weight"] = weight_sum
        denominator = weight_sum.where(weight_sum > 0)
        out["forecast"] = return_sum / denominator
        out["win_rate"] = win_sum / denominator
        out["eligible"] = (
            out["analog_count"].ge(params.minimum_analogs)
            & out["forecast"].gt(params.forecast_threshold)
            & out["win_rate"].ge(params.minimum_win_rate)
            & out["atr"].gt(0)
        )
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_decision = -1
        high_water = np.nan
        trailing_stop = -np.inf
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                # Entry fills this bar after the preceding bar's decision.
                # Only closes observed during the trade update its high-water mark.
                if valid:
                    high_water = close if not np.isfinite(high_water) else max(high_water, close)
                    trailing_stop = max(trailing_stop, high_water - params.trail_atr * atr)
                expired = i - entry_decision >= params.holding_bars
                stopped = valid and close <= trailing_stop
                if not valid or stopped or expired:
                    active = False
                    high_water = np.nan
                    trailing_stop = -np.inf
                else:
                    decisions[i] = 1
            elif i >= warmup - 1 and valid and confirmed[i]:
                active = True
                entry_decision = i
                high_water = np.nan
                trailing_stop = -np.inf
                decisions[i] = 1

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
