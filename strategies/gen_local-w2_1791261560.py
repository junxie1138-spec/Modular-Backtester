from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 50
    dwell_window: int = 12
    dwell_fraction: float = 0.75
    entry_z: float = 1.0
    atr_window: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791261560"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.ma_window + params.dwell_window,
                   params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.ma_window < 2 or params.dwell_window < 1
                or params.atr_window < 1 or params.hold_bars < 1
                or not 0 < params.dwell_fraction <= 1
                or not np.isfinite(params.entry_z) or params.entry_z <= 0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["dispersion"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0, np.nan)
        out["regime_ma"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["tr"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window).mean()
        submerged = (out["z"] <= -params.entry_z).astype(float)
        submerged = submerged.where(out["z"].notna())
        out["dwell"] = submerged.rolling(params.dwell_window).mean().shift(1)
        out["release"] = (
            (out["z"] > -params.entry_z)
            & (out["z"].shift(1) <= -params.entry_z)
            & (out["dwell"] >= params.dwell_fraction)
        ).fillna(False)
        out["entry"] = (
            out["release"] & (close > out["regime_ma"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["regime_ma"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = (np.isfinite(close[i]) and np.isfinite(atr[i])
                     and atr[i] > 0 and np.isfinite(regime[i]))
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if (not valid or close[i] <= stop or close[i] <= regime[i]
                        or age >= params.hold_bars):
                    active = False
                signal[i] = int(active)
            elif valid and entry[i]:
                active = True
                age = 0
                # The first in-trade close arrives after the next-bar fill.
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 110., 105., 104., 103.]},
                        index=index)
    derived = pd.DataFrame(
        {"atr": [1., 1., 1., 10., 10., 10.],
         "regime_ma": 90., "entry": [True, False, False, False, False, False]},
        index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    params.hold_bars = 2
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
