from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    box_bars: int = 6
    peak_lookback: int = 30
    compression_atr: float = 2.5
    depth_trigger: float = 0.03
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791301206"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.box_bars, params.peak_lookback, 20) + 2

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        previous = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(20, min_periods=20).mean().replace(0, np.nan)
        upper = data["high"].rolling(params.box_bars, min_periods=params.box_bars).max()
        lower = data["low"].rolling(params.box_bars, min_periods=params.box_bars).min()
        peak = data["close"].rolling(params.peak_lookback, min_periods=params.peak_lookback).max()

        out = pd.DataFrame(index=data.index)
        out["box_high"] = upper.shift(2).fillna(data["close"])
        out["box_low"] = lower.shift(2).fillna(data["close"])
        out["compression"] = ((upper - lower) / atr).shift(2).replace([np.inf, -np.inf], np.nan).fillna(params.compression_atr + 1.0)
        out["drawdown"] = (1.0 - data["close"] / peak).shift(2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"]
        compressed = indicators["compression"] <= params.compression_atr
        long_entry = (
            compressed
            & (indicators["drawdown"] >= params.depth_trigger)
            & (close > indicators["box_high"])
            & (close.shift(1) > indicators["box_high"])
        ).fillna(False).to_numpy(dtype=bool)
        short_entry = (
            compressed
            & (indicators["drawdown"] <= params.depth_trigger / 2.0)
            & (close < indicators["box_low"])
            & (close.shift(1) < indicators["box_low"])
        ).fillna(False).to_numpy(dtype=bool)

        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        position = 0
        exit_at = -1
        for i in range(len(df)):
            if i == exit_at:
                position = 0
            elif position == 0:
                if long_entry[i]:
                    position = 1
                    exit_at = i + params.hold_bars
                elif short_entry[i]:
                    position = -1
                    exit_at = i + params.hold_bars
            df.iat[i, 0] = position

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"close": [99, 101, 102, 102, 102, 102]}, index=index)
    levels = pd.DataFrame({"box_high": 100.0, "box_low": 95.0, "compression": 1.0, "drawdown": 0.04}, index=index)
    signals = GeneratedStrategy().generate_signals(prices, levels, StrategyContext("SPY", "1d", 0), StrategyParams()).data["signal"]
    assert signals.iloc[2:6].tolist() == [0, 1, 1, 0]
