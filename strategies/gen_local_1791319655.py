from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    box_bars: int = 3
    percentile_window: int = 80
    compression_quantile: float = 0.20
    atr_bars: int = 14
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791319655"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.percentile_window + params.box_bars + 1,
                   params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (min(params.box_bars, params.percentile_window, params.atr_bars) < 1
                or not 0 < params.compression_quantile < 1
                or params.trail_atr <= 0 or not 3 <= params.hold_bars <= 5):
            raise ValueError("Invalid lookbacks, quantile, trailing distance, or horizon")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out["range_fraction"] = (data["high"] - data["low"]) / previous_close
        out["compression"] = out["range_fraction"].rolling(params.box_bars).mean()
        out["compression_cutoff"] = out["compression"].rolling(
            params.percentile_window).quantile(params.compression_quantile).shift(1)
        out["expansion_cutoff"] = out["range_fraction"].rolling(
            params.percentile_window).quantile(0.65).shift(1)
        out["box_high"] = data["high"].rolling(params.box_bars).max().shift(1)
        out["box_low"] = data["low"].rolling(params.box_bars).min().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["compressed_before"] = (
            out["compression"].shift(1) <= out["compression_cutoff"].shift(1))
        out["expanded"] = out["range_fraction"] > out["expansion_cutoff"]
        out["inside"] = ((data["close"] > out["box_low"])
                         & (data["close"] < out["box_high"]))
        out["ready"] = (
            np.isfinite(out[["range_fraction", "compression_cutoff",
                             "expansion_cutoff", "box_high", "box_low", "atr"]]).all(axis=1)
            & (out["atr"] > 0)
            & np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1))
        setup = out["ready"] & out["compressed_before"] & out["expanded"] & out["inside"]
        out["long_entry"] = (setup & (data["low"] < out["box_low"])
                              & (data["high"] <= out["box_high"])
                              & (data["close"] > data["open"]))
        out["short_entry"] = (setup & (data["high"] > out["box_high"])
                               & (data["low"] >= out["box_low"])
                               & (data["close"] < data["open"]))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        extreme = 0.0
        stop = 0.0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].to_numpy(dtype=bool)
        longs = indicators["long_entry"].to_numpy(dtype=bool)
        shorts = indicators["short_entry"].to_numpy(dtype=bool)
        for i in range(len(data)):
            if position:
                if not ready[i]:
                    position = 0
                else:
                    # Entry fills at this bar's open; exclude the decision close.
                    if i == entry_bar:
                        extreme = max(position * opens[i], position * closes[i])
                        stop = extreme - params.trail_atr * atrs[i]
                    else:
                        extreme = max(extreme, position * closes[i])
                        stop = max(stop, extreme - params.trail_atr * atrs[i])
                    # Signed prices mirror the high-water stop for shorts.
                    if (position * closes[i] <= stop
                            or i - entry_bar + 1 >= params.hold_bars):
                        position = 0
                signals[i] = position
                continue
            if ready[i] and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                entry_bar = i + 1
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100, 102, 104, 103, 101, 100],
                         "close": [100, 102, 104, 103, 101, 100]}, index=index)
    ind = pd.DataFrame({"atr": [1, 1, 1, 10, 10, 1], "ready": True,
                        "long_entry": [True, False, False, False, False, False],
                        "short_entry": False}, index=index)
    result = strategy.generate_signals(
        data, ind, None, Params(trail_atr=2.0, hold_bars=5)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    data["open"] = data["close"] = [100, 99, 98, 98, 98, 98]
    ind["atr"] = 1.0
    ind["long_entry"] = False
    ind["short_entry"] = [True, False, False, False, False, False]
    result = strategy.generate_signals(
        data, ind, None, Params(trail_atr=2.0, hold_bars=3)).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
