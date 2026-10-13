from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePursuitParams:
    innovation_threshold: float = 0.20
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[RangePursuitParams]):
    strategy_id = "gen_local-w2_1791290879"

    @classmethod
    def params_type(cls) -> type[RangePursuitParams]:
        return RangePursuitParams

    @staticmethod
    def warmup_bars(params: RangePursuitParams) -> int:
        return 14

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangePursuitParams) -> pd.DataFrame:
        length = 12
        midpoint = (data["high"] + data["low"]) / 2.0
        x = np.arange(length, dtype=float)
        centered_x = x - x.mean()
        values = midpoint.fillna(0.0).to_numpy(dtype=float)
        weighted = (
            np.convolve(values, centered_x[::-1], mode="full")[:len(values)]
            if len(values)
            else np.empty(0, dtype=float)
        )
        valid = midpoint.rolling(length, min_periods=length).count().eq(length)
        slope = pd.Series(weighted, index=data.index).div(
            float(np.square(centered_x).sum())
        ).where(valid)
        equilibrium = midpoint.rolling(length, min_periods=length).mean() + slope * x.mean()
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        scale = true_range.rolling(length, min_periods=length).mean().replace(0.0, np.nan)
        slope_forecast = 2.0 * slope.shift(1) - slope.shift(2)
        innovation = (slope - slope_forecast) / (scale / length)
        relative_position = (data["close"] - equilibrium) / scale
        return pd.DataFrame(
            {
                "midpoint": midpoint,
                "slope": slope,
                "equilibrium": equilibrium,
                "true_range": true_range,
                "scale": scale,
                "slope_forecast": slope_forecast,
                "innovation": innovation,
                "relative_position": relative_position,
            },
            index=data.index,
        ).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangePursuitParams,
    ) -> SignalFrame:
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0.0:
            raise ValueError("innovation_threshold must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be finite and positive")

        relative = indicators["relative_position"]
        innovation = indicators["innovation"]
        slope = indicators["slope"]
        long_entry = (
            relative.between(-0.75, -0.05)
            & slope.gt(0.0)
            & innovation.gt(params.innovation_threshold)
        ).fillna(False)
        short_entry = (
            relative.between(0.05, 0.75)
            & slope.lt(0.0)
            & innovation.lt(-params.innovation_threshold)
        ).fillna(False)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        bars_held = 0

        for i in range(len(data)):
            if position != 0:
                # A decision from the previous close fills at today's open.
                if bars_held == 0:
                    entry_price = opens[i]
                bars_held += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = (
                    position * (closes[i] / entry_price - 1.0)
                    if valid_entry and np.isfinite(closes[i])
                    else np.nan
                )
                if not valid_entry or gain >= params.profit_target or bars_held >= 4:
                    position = 0
                    entry_price = np.nan
                    bars_held = 0
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidates[i])
                entry_price = np.nan
                bars_held = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
