from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_period: int = 3
    volatility_window: int = 20
    acceleration_min: float = 0.5
    floor_lift_min: float = 0.2
    hold_bars: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790850935"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.roc_period + 2, params.volatility_window + 1, 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.roc_period < 1 or params.volatility_window < 2:
            raise ValueError("Lookbacks must be positive; volatility_window must be at least two.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not np.isfinite(params.size) or not 0 < params.size <= 1:
            raise ValueError("size must be finite and in (0, 1].")
        if not np.isfinite(params.acceleration_min) or params.acceleration_min < 0:
            raise ValueError("acceleration_min must be finite and nonnegative.")
        if not np.isfinite(params.floor_lift_min) or not 0 < params.floor_lift_min < 0.5:
            raise ValueError("floor_lift_min must be in (0, 0.5).")

        out = pd.DataFrame(index=data.index)
        out["daily_return"] = data["close"].pct_change(fill_method=None)
        out["roc"] = data["close"].pct_change(params.roc_period, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["volatility"] = out["daily_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["scaled_acceleration"] = out["acceleration"] / out["volatility"].replace(0, np.nan)
        out["previous_low"] = data["low"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_range"] = (out["previous_high"] - out["previous_low"]).where(
            out["previous_high"] > out["previous_low"]
        )
        out["floor_lift"] = (data["low"] - out["previous_low"]) / out["previous_range"]
        out["overlap_high"] = np.minimum(data["high"], out["previous_high"])
        out["overlap_low"] = np.maximum(data["low"], out["previous_low"])
        out["overlap_fraction"] = (
            out["overlap_high"] - out["overlap_low"]
        ).clip(lower=0) / out["previous_range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["accelerating"] = (out["roc"] > 0) & (
            out["scaled_acceleration"] > params.acceleration_min
        )
        out["floor_acceptance"] = (out["floor_lift"] >= params.floor_lift_min) & (
            out["overlap_fraction"] >= 0.5
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        entries = (
            indicators["accelerating"].fillna(False)
            & indicators["floor_acceptance"].fillna(False)
        ).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if i < exit_bar:
                raw[i] = 1
            elif i == exit_bar:
                # Preserve the scheduled flat bar even if both primitives agree.
                continue
            elif enter:
                exit_bar = i + params.hold_bars
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"accelerating": True, "floor_acceptance": True}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    indicators.loc[index[1:], "accelerating"] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    indicators["floor_acceptance"] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].eq(0).all()
    assert result["size"].gt(0).all()
