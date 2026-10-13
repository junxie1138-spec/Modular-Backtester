from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 63
    tail_cutoff: float = 0.15
    memory_floor: float = 0.05
    hold_bars: int = 18

    def __post_init__(self):
        if self.lookback < 16 or self.hold_bars < 1:
            raise ValueError("lookback must be >= 16 and hold_bars >= 1")
        if not 0.0 < self.tail_cutoff < 1.0:
            raise ValueError("tail_cutoff must lie between zero and one")
        if not 0.0 <= self.memory_floor < 1.0:
            raise ValueError("memory_floor must lie in [0, 1)")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791234212"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params):
        return int(params.lookback + 4)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff().fillna(0.0)
        out["direction"] = np.sign(out["change"])
        out["previous_direction"] = out["direction"].shift(1, fill_value=0.0)
        out["up"] = out["direction"].eq(1)
        out["valid_pair"] = out["direction"].ne(0) & out["previous_direction"].ne(0)
        out["up_total"] = out["up"].astype(int).cumsum()
        out["reset_total"] = out["up_total"].where(~out["up"], 0).cummax()
        out["streak"] = out["up_total"] - out["reset_total"]
        for label, previous in (("up", 1), ("down", -1)):
            out[label + "_exposure"] = (
                out["valid_pair"] & out["previous_direction"].eq(previous)
            ).astype(float)
            out[label + "_success"] = out[label + "_exposure"] * out["up"]
            for suffix in ("exposure", "success"):
                name = label + "_" + suffix
                out[name + "_count"] = (
                    out[name].rolling(params.lookback, min_periods=params.lookback)
                    .sum().shift(1).fillna(0.0)
                )
            out["p_after_" + label] = (
                (out[label + "_success_count"] + 1.0)
                / (out[label + "_exposure_count"] + 2.0)
            )
        out["memory"] = out["p_after_up"] - out["p_after_down"]
        out["streak_tail"] = np.power(
            out["p_after_up"], (out["streak"] - 1).clip(lower=0)
        )
        out["ready"] = np.arange(len(data)) >= self.warmup_bars(params)
        out["setup"] = (
            out["ready"]
            & out["up_exposure_count"].ge(8)
            & out["down_exposure_count"].ge(8)
            & out["p_after_up"].gt(0.5)
            & out["memory"].ge(params.memory_floor)
            & out["streak"].ge(2)
            & out["streak_tail"].le(params.tail_cutoff)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        out["entry"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        exit_bar = -1
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                raw[i] = 1
            elif entries[i]:
                raw[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=100, freq="B")
    close = pd.Series(100.0 + np.arange(100) * 0.1, index=index)
    data = pd.DataFrame({name: close for name in ("open", "high", "low", "close")})
    data["volume"] = 1000000.0
    derived = strategy.indicators(data, params)
    assert not derived.isna().any().any()
    assert derived["confirmed"].equals(
        derived["setup"] & derived["setup"].shift(1, fill_value=False)
    )
    pd.testing.assert_frame_equal(
        strategy.indicators(data.iloc[:80], params), derived.iloc[:80]
    )
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.iloc[[4, 6, 22, 40], 0] = True
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, entries, ctx, params).data
    expected = np.zeros(100, dtype=int)
    expected[5:23] = 1
    expected[41:59] = 1
    assert np.array_equal(signals["signal"].to_numpy(), expected)
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
