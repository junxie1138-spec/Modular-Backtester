from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChannelReleaseParams:
    range_bars: int = 5
    streak_bars: int = 3
    edge: float = 0.8
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ChannelReleaseParams]):
    strategy_id = "gen_local_1790941001"

    @classmethod
    def params_type(cls) -> type[ChannelReleaseParams]:
        return ChannelReleaseParams

    def warmup_bars(self, params: ChannelReleaseParams) -> int:
        if params.range_bars < 2 or params.streak_bars < 1:
            raise ValueError("Lookbacks must be positive; range_bars must be at least two")
        if params.range_bars + params.streak_bars > 10:
            raise ValueError("Combined lookbacks must not exceed ten bars")
        if not 0.5 < params.edge < 1.0 or params.hold_bars < 1:
            raise ValueError("Invalid edge or holding period")
        return int(params.range_bars + params.streak_bars)

    def indicators(self, data: pd.DataFrame, params: ChannelReleaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max()
        out["channel_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min()
        width = (out["channel_high"] - out["channel_low"]).where(lambda s: s > 0)
        out["relative_position"] = ((data["close"] - out["channel_low"]) / width).replace([np.inf, -np.inf], np.nan)
        upper = out["relative_position"].ge(params.edge)
        lower = out["relative_position"].le(1.0 - params.edge)
        upper_run = pd.Series(True, index=data.index)
        lower_run = pd.Series(True, index=data.index)
        out["upper_streak"] = 0
        out["lower_streak"] = 0
        for offset in range(params.streak_bars):
            upper_run = upper_run & upper.shift(offset, fill_value=False)
            lower_run = lower_run & lower.shift(offset, fill_value=False)
            out["upper_streak"] += upper_run.astype(int)
            out["lower_streak"] += lower_run.astype(int)
        position = out["relative_position"]
        out["long_entry"] = (
            out["lower_streak"].shift(1).ge(params.streak_bars)
            & position.gt(1.0 - params.edge)
            & position.lt(0.5)
        ).fillna(False)
        out["short_entry"] = (
            out["upper_streak"].shift(1).ge(params.streak_bars)
            & position.lt(params.edge)
            & position.gt(0.5)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChannelReleaseParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            elif long_entry[bar] or short_entry[bar]:
                position = 1 if long_entry[bar] else -1
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = ChannelReleaseParams(hold_bars=15)
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"long_entry": False, "short_entry": False}, index=index)
    entries.iloc[0, 0] = True
    entries.iloc[2, 1] = True
    entries.iloc[19, 1] = True
    result = strategy.generate_signals(data, entries, None, params).data
    expected = np.zeros(40, dtype=int)
    expected[1:16] = 1
    expected[20:35] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
    derived = strategy.indicators(data, params)
    assert not derived["long_entry"].any()
    assert not derived["short_entry"].any()
    assert strategy.warmup_bars(params) <= 10
