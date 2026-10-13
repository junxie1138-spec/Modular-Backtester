from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EscapeParams:
    streak_length: int = 2
    breakout_lookback: int = 12
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[EscapeParams]):
    strategy_id = "gen_local_1790866042"

    @classmethod
    def params_type(cls) -> type[EscapeParams]:
        return EscapeParams

    def warmup_bars(self, params: EscapeParams) -> int:
        return int(max(200, params.breakout_lookback + 1, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: EscapeParams) -> pd.DataFrame:
        if min(params.streak_length, params.breakout_lookback, params.holding_bars) < 1:
            raise ValueError("All lookbacks and holding_bars must be positive")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["ceiling"] = data["high"].shift(1).rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max()
        out["floor"] = data["low"].shift(1).rolling(params.breakout_lookback, min_periods=params.breakout_lookback).min()
        out["up_escape"] = ((data["close"] > data["high"].shift(1)) & (data["close"] > data["open"])).fillna(False)
        out["down_escape"] = ((data["close"] < data["low"].shift(1)) & (data["close"] < data["open"])).fillna(False)
        for direction in ("up", "down"):
            escape = out[direction + "_escape"]
            out[direction + "_streak"] = escape.astype(int).groupby((~escape).cumsum()).cumsum()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EscapeParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        ready = indicators[["ma200", "ceiling", "floor", "up_streak", "down_streak"]].notna().all(axis=1)
        long_entry = ready & (indicators["up_streak"] == params.streak_length) & (data["close"] > indicators["ceiling"]) & (data["close"] > indicators["ma200"])
        short_entry = ready & (indicators["down_streak"] == params.streak_length) & (data["close"] < indicators["floor"]) & (data["close"] < indicators["ma200"])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar == exit_at:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_at = bar + params.holding_bars
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = EscapeParams()
    index = pd.date_range("2020-01-01", periods=14, freq="D")
    data = pd.DataFrame({"close": [101.0] * 7 + [99.0] * 7}, index=index)
    indicators = pd.DataFrame({"ma200": 100.0, "ceiling": 100.0, "floor": 100.0, "up_streak": 0, "down_streak": 0}, index=index)
    indicators.loc[index[1], "up_streak"] = 2
    indicators.loc[index[3], "down_streak"] = 2
    indicators.loc[index[7], "down_streak"] = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
