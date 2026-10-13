from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapClearanceParams:
    range_bars: int = 12
    atr_bars: int = 14
    max_range_atr: float = 3.0
    min_gap_atr: float = 0.25
    trail_atr: float = 2.0
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapClearanceParams]):
    strategy_id = "gen_local_1790905903"

    @classmethod
    def params_type(cls):
        return GapClearanceParams

    def warmup_bars(self, params):
        return int(max(200, params.range_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        if params.range_bars < 2 or params.atr_bars < 1 or params.max_hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (params.max_range_atr, params.min_gap_atr, params.trail_atr)):
            raise ValueError("ATR thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["ma200"] = data["close"].rolling(200).mean()
        out["ceiling"] = data["high"].rolling(params.range_bars).max().shift(1)
        out["floor"] = data["low"].rolling(params.range_bars).min().shift(1)
        denominator = out["prior_atr"].where(out["prior_atr"] > 0)
        out["range_atr"] = (out["ceiling"] - out["floor"]) / denominator
        out["gap_atr"] = (data["open"] - previous_close) / denominator
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["range_atr"] <= params.max_range_atr)
            & (out["gap_atr"] >= params.min_gap_atr)
            & (data["open"] > out["ceiling"])
            & (data["low"] > out["ceiling"])
            & (data["close"] >= data["open"])
            & (data["close"] > out["ma200"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or close[i] <= regime[i] or age >= params.max_hold_bars:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                # Entry fills next bar; start the closing high-water mark then.
                high_water = stop = -np.inf
            target[i] = int(active)
        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [999., 100., 105., 104., 102., 101.]}, index=index)
    indicators = pd.DataFrame(
        {"atr": [1., 1., 1., 10., 1., 1.],
         "ma200": 50., "entry": [True, False, False, False, False, False]},
        index=index
    )
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, GapClearanceParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    result = strategy.generate_signals(data, indicators, ctx, GapClearanceParams(max_hold_bars=2)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
