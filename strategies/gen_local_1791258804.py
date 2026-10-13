from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    atr_period: int = 14
    volume_mult: float = 1.1
    saturation: float = 0.75
    atr_mult: float = 1.0
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791258804"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.lookback + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; lookback must be at least two.")
        if not (0.45 < params.saturation < 1.0):
            raise ValueError("saturation must be between 0.45 and 1.")
        if params.volume_mult <= 0 or params.atr_mult <= 0 or params.hold_days not in (1, 2):
            raise ValueError("Multipliers must be positive and hold_days must be one or two.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["up_volume"] = data["volume"].where(data["close"] > out["previous_close"], 0.0)
        out["up_volume"] = out["up_volume"].where(out["previous_close"].notna())
        out["volume_sum"] = data["volume"].rolling(params.lookback).sum()
        out["up_volume_sum"] = out["up_volume"].rolling(params.lookback).sum()
        out["up_share"] = (out["up_volume_sum"] / out["volume_sum"].replace(0, np.nan)).shift(1)
        out["volume_reference"] = data["volume"].rolling(params.lookback).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_reference"].replace(0, np.nan)
        out["close_location"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["regime"] = np.select([
            out["up_share"].le(0.45),
            out["up_share"].gt(0.45) & out["up_share"].lt(params.saturation),
        ], [1, 2], default=0)
        out["recovery"] = (
            out["regime"].eq(1)
            & data["volume"].gt(data["volume"].shift(1))
            & data["close"].gt(data["open"])
            & out["close_location"].ge(0.7)
        )
        out["expansion"] = (
            out["regime"].eq(2)
            & data["close"].gt(out["previous_high"])
            & out["close_location"].ge(0.6)
        )
        out["eligible"] = (
            (out["recovery"] | out["expansion"])
            & data["close"].gt(out["previous_close"])
            & out["relative_volume"].ge(params.volume_mult)
            & out["atr"].gt(0)
        ).fillna(False)
        out["confirmed"] = (
            out["eligible"]
            & out["eligible"].shift(1, fill_value=False)
            & out["regime"].eq(out["regime"].shift(1))
        ).fillna(False)
        out["entry"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest = -np.inf
        stop = -np.inf
        for i, (close, atr, entry) in enumerate(zip(
            data["close"].to_numpy(),
            indicators["atr"].to_numpy(),
            indicators["entry"].fillna(False).to_numpy(),
        )):
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                active = False
                age = 0
                highest = stop = -np.inf
                continue
            if active:
                age += 1
                highest = max(highest, close)
                stop = max(stop, highest - params.atr_mult * atr)
                if close <= stop or age >= params.hold_days:
                    active = False
            elif entry:
                active = True
                age = 0
                # Entry fills next bar; start recording in-trade closes then.
                highest = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 103., 103., 101.5, 102.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 1., 1., 10., 1.],
        "entry": [False, True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    # Isolate the stop from expiry: rising ATR must not lower the stop.
    result = strategy.generate_signals(data, indicators, None, Params(hold_days=10)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
