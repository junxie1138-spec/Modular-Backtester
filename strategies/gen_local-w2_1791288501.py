from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    lookback: int = 15
    streak_bars: int = 2
    pressure_bars: int = 4
    pressure_threshold: float = 0.35
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.lookback, self.streak_bars, self.pressure_bars, self.hold_bars) < 1:
            raise ValueError("Bar counts must be positive.")
        if not 0.0 < self.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local-w2_1791288501"

    @classmethod
    def params_type(cls):
        return BreakoutParams

    @staticmethod
    def warmup_bars(params: BreakoutParams) -> int:
        return int(max(params.lookback + params.streak_bars, params.pressure_bars))

    @staticmethod
    def indicators(data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["upper"] = data["high"].shift(1).rolling(params.lookback, min_periods=params.lookback).max()
        out["lower"] = data["low"].shift(1).rolling(params.lookback, min_periods=params.lookback).min()
        out["body"] = data["close"] - data["open"]
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["body_sum"] = out["body"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["range_sum"] = out["range"].rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        out["pressure"] = (out["body_sum"] / out["range_sum"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["up_break"] = data["close"].gt(out["upper"]).fillna(False)
        out["down_break"] = data["close"].lt(out["lower"]).fillna(False)
        for direction in ("up", "down"):
            breaks = out[direction + "_break"]
            out[direction + "_streak"] = breaks.astype(int).groupby((~breaks).cumsum()).cumsum()
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        long_entry = indicators["up_streak"].eq(params.streak_bars) & indicators["pressure"].ge(params.pressure_threshold)
        short_entry = indicators["down_streak"].eq(params.streak_bars) & indicators["pressure"].le(-params.pressure_threshold)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position and i == exit_bar:
                position = 0
                continue
            if position == 0 and entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = BreakoutParams()
    index = pd.date_range("2020-01-01", periods=32)
    price = pd.Series(np.arange(32, dtype=float) + 100.0, index=index)
    data = pd.DataFrame({"open": price - 0.5, "high": price + 0.1, "low": price - 0.9, "close": price, "volume": 1000.0})
    derived = GeneratedStrategy.indicators(data, params)
    assert derived["up_streak"].iloc[:params.lookback].eq(0).all()
    assert derived["up_streak"].iloc[params.lookback + 1] == 2
    assert GeneratedStrategy.indicators(data.iloc[:24], params).equals(derived.iloc[:24])
    forced = pd.DataFrame({"up_streak": 0, "down_streak": 0, "pressure": 0.0}, index=index)
    forced.loc[index[2], ["up_streak", "pressure"]] = [2, 0.5]
    forced.loc[index[4], ["down_streak", "pressure"]] = [2, -0.5]
    forced.loc[index[12], ["down_streak", "pressure"]] = [2, -0.5]
    result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    expected = np.zeros(32, dtype=int)
    expected[3:11] = 1
    expected[13:21] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
