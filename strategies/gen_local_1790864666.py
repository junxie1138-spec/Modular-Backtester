from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    history_bars: int = 189
    volatility_bars: int = 20
    shape_bars: int = 5
    neighbors: int = 7
    forecast_floor: float = 0.015
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = "gen_local_1790864666"

    @classmethod
    def params_type(cls) -> type[GapAnalogParams]:
        return GapAnalogParams

    def warmup_bars(self, params: GapAnalogParams) -> int:
        return int(params.history_bars + max(params.volatility_bars, params.shape_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: GapAnalogParams) -> pd.DataFrame:
        if (params.volatility_bars < 2 or params.shape_bars < 2
                or params.hold_bars < 1 or params.neighbors < 1
                or params.history_bars < params.hold_bars + params.neighbors + 1
                or params.profit_target <= 0):
            raise ValueError("Invalid gap analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        gap = data["open"].div(close.shift(1)).sub(1)
        gap = gap.replace([np.inf, -np.inf], np.nan)
        scale = gap.rolling(params.volatility_bars, min_periods=params.volatility_bars).std()
        scale = scale.where(scale > 1e-8)
        out["gap"] = gap
        out["gap_scale"] = scale
        out["current_gap_shape"] = gap.div(scale)
        out["previous_gap_shape"] = gap.shift(1).div(scale.shift(1))
        out["cumulative_gap"] = gap.rolling(params.shape_bars, min_periods=params.shape_bars).sum()
        out["cumulative_gap_shape"] = out["cumulative_gap"].div(scale * np.sqrt(params.shape_bars))
        out["intraday_return"] = close.div(data["open"].where(data["open"] > 0)).sub(1)
        features = out[["current_gap_shape", "previous_gap_shape", "cumulative_gap_shape"]]
        current = features.to_numpy(dtype=float)
        distances = []
        outcomes = []
        for lag in range(params.hold_bars + 2, params.history_bars + 1):
            historical = features.shift(lag).to_numpy(dtype=float)
            distance = np.square(current - historical).sum(axis=1)
            outcome = close.shift(lag - params.hold_bars).div(close.shift(lag)).sub(1).to_numpy(dtype=float)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(np.where(valid, distance, np.inf))
            outcomes.append(np.where(valid, outcome, np.nan))
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argsort(distance_matrix, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        out["analog_count"] = np.isfinite(distance_matrix).sum(axis=1)
        out["analog_forecast"] = np.where(complete, selected_outcomes.mean(axis=1), np.nan)
        out["confirmation"] = (
            out["analog_forecast"].ge(params.forecast_floor)
            & out["cumulative_gap"].lt(0)
            & out["intraday_return"].gt(0)
        ).fillna(False)
        out["entry_ready"] = (
            out["confirmation"] & out["confirmation"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapAnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        holding = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if holding:
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                time_hit = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_hit:
                    holding = False
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                holding = True
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
