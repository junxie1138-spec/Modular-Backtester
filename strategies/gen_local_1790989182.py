from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalExpansionParams:
    fractional_order: float = 0.65
    fractional_window: int = 40
    normalization_window: int = 30
    movement_window: int = 8
    fractional_threshold: float = 0.7
    movement_threshold: float = 0.12
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalExpansionParams]):
    strategy_id = "gen_local_1790989182"

    @classmethod
    def params_type(cls) -> type[FractionalExpansionParams]:
        return FractionalExpansionParams

    def warmup_bars(self, params: FractionalExpansionParams) -> int:
        return int(max(
            params.fractional_window + params.normalization_window,
            params.movement_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalExpansionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag

        fractional_close = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional_close = fractional_close + weight * close.shift(lag)
        out["fractional_close"] = fractional_close
        normalization = out["fractional_close"].rolling(
            params.normalization_window, min_periods=params.normalization_window
        )
        out["fractional_mean"] = normalization.mean()
        out["fractional_std"] = normalization.std(ddof=0)
        out["fractional_score"] = (
            (out["fractional_close"] - out["fractional_mean"])
            / out["fractional_std"].replace(0.0, np.nan)
        )

        out["high_change"] = data["high"].diff()
        out["low_decline"] = -data["low"].diff()
        valid_moves = out["high_change"].notna() & out["low_decline"].notna()
        out["up_movement"] = out["high_change"].where(
            (out["high_change"] > out["low_decline"]) & (out["high_change"] > 0.0),
            0.0,
        ).where(valid_moves)
        out["down_movement"] = out["low_decline"].where(
            (out["low_decline"] > out["high_change"]) & (out["low_decline"] > 0.0),
            0.0,
        ).where(valid_moves)
        out["previous_close"] = close.shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap_range"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap_range"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[
            ["daily_range", "high_gap_range", "low_gap_range"]
        ].max(axis=1).where(out["previous_close"].notna())
        out["up_sum"] = out["up_movement"].rolling(
            params.movement_window, min_periods=params.movement_window
        ).sum()
        out["down_sum"] = out["down_movement"].rolling(
            params.movement_window, min_periods=params.movement_window
        ).sum()
        out["range_sum"] = out["true_range"].rolling(
            params.movement_window, min_periods=params.movement_window
        ).sum()
        out["directional_expansion"] = (
            (out["up_sum"] - out["down_sum"])
            / out["range_sum"].replace(0.0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalExpansionParams,
    ) -> SignalFrame:
        fractional = indicators["fractional_score"]
        expansion = indicators["directional_expansion"]
        valid = fractional.notna() & expansion.notna()
        long_entry = valid & (fractional > params.fractional_threshold) & (
            expansion > params.movement_threshold
        )
        short_entry = valid & (fractional < -params.fractional_threshold) & (
            expansion < -params.movement_threshold
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        warmup = self.warmup_bars(params)
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
                    entry_bar = -1
                    # Preserve a flat exit bar, even when another entry qualifies.
                    continue
                signals[bar] = position
            elif bar >= warmup - 1 and candidates[bar] != 0:
                position = int(candidates[bar])
                entry_bar = bar
                signals[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
