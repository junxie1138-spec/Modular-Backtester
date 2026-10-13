from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRangeParams:
    atr_bars: int = 20
    spike_ratio: float = 2.5
    refractory_bars: int = 5
    anchor_life: int = 40
    entry_level: float = 0.75
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockRangeParams]):
    strategy_id = "gen_local_1791230429"

    @classmethod
    def params_type(cls) -> type[ShockRangeParams]:
        return ShockRangeParams

    def warmup_bars(self, params: ShockRangeParams) -> int:
        return int(max(200, params.atr_bars + 2))

    def indicators(self, data: pd.DataFrame, params: ShockRangeParams) -> pd.DataFrame:
        if params.atr_bars < 2 or params.spike_ratio <= 1:
            raise ValueError("Require atr_bars >= 2 and spike_ratio > 1")
        if params.refractory_bars < 1 or params.anchor_life < params.refractory_bars:
            raise ValueError("Invalid anchor timing")
        if not 0.5 < params.entry_level < 1 or params.holding_bars < 1:
            raise ValueError("Invalid entry level or holding period")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["shock_ratio"] = out["true_range"].div(out["atr"].shift(1).replace(0.0, np.nan))
        out["spike"] = out["shock_ratio"].ge(params.spike_ratio) & out["shock_ratio"].shift(1).lt(params.spike_ratio)
        out["anchor_high"] = data["high"].where(out["spike"]).ffill()
        out["anchor_low"] = data["low"].where(out["spike"]).ffill()
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["anchor_bar"] = out["bar_number"].where(out["spike"]).ffill()
        out["anchor_age"] = out["bar_number"] - out["anchor_bar"]
        out["anchor_width"] = (out["anchor_high"] - out["anchor_low"]).where(lambda s: s > 0)
        out["relative_position"] = (data["close"] - out["anchor_low"]).div(out["anchor_width"])
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["eligible"] = (
            out["anchor_age"].between(params.refractory_bars, params.anchor_life)
            & out["bar_number"].ge(self.warmup_bars(params) - 1)
        )
        position = out["relative_position"]
        lower_level = 1.0 - params.entry_level
        long_entry = (
            out["eligible"]
            & data["close"].gt(out["ma200"])
            & position.gt(params.entry_level)
            & position.le(1.0)
            & position.shift(1).le(params.entry_level)
        )
        short_entry = (
            out["eligible"]
            & data["close"].lt(out["ma200"])
            & position.lt(lower_level)
            & position.ge(0.0)
            & position.shift(1).ge(lower_level)
        )
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockRangeParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(candidates):
            if position != 0 and bar == exit_bar:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = bar + params.holding_bars
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"candidate": [0, 1, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, ShockRangeParams(holding_bars=3))
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
