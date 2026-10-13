from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRepairParams:
    range_window: int = 20
    drawdown_window: int = 15
    atr_window: int = 14
    shock_multiple: float = 1.4
    repair_fraction: float = 0.15
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[RangeRepairParams]):
    strategy_id = "gen_local_1791321392"

    @classmethod
    def params_type(cls):
        return RangeRepairParams

    def warmup_bars(self, params):
        return max(200, params.range_window + 2,
                   params.drawdown_window + 1, params.atr_window + 1)

    def indicators(self, data, params):
        if min(params.range_window, params.drawdown_window, params.atr_window) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not (params.shock_multiple > 0 and 0 < params.repair_fraction < 1
                and params.trail_atr > 0):
            raise ValueError("Invalid range or trailing-stop parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        prev_close = close.shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_baseline"] = out["range"].rolling(params.range_window).mean().shift(2)
        out["ma200"] = close.rolling(200).mean()
        out["true_range"] = pd.concat([
            out["range"], (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["prior_peak"] = close.rolling(params.drawdown_window).max().shift(1)
        prior_range = out["range"].shift(1).replace(0, np.nan)
        out["lower_repair"] = (data["low"] - data["low"].shift(1)) / prior_range
        out["upper_repair"] = (data["high"].shift(1) - data["high"]) / prior_range
        out["inside"] = ((out["lower_repair"] > 0) & (out["upper_repair"] > 0))
        out["shock"] = out["range"].shift(1) > params.shock_multiple * out["range_baseline"]
        out["drawdown"] = out["prior_peak"] - prev_close
        out["ready"] = np.isfinite(out[[
            "range_baseline", "ma200", "atr", "prior_peak",
            "lower_repair", "upper_repair", "drawdown"
        ]]).all(axis=1) & (out["atr"] > 0)
        common = out["ready"] & out["inside"] & out["shock"] & (out["drawdown"] > 0.5 * out["atr"].shift(1))
        out["long_entry"] = (common & (close > out["ma200"])
            & (prev_close < close.shift(2)) & (close > prev_close)
            & ((out["lower_repair"] - out["upper_repair"]) >= params.repair_fraction))
        out["short_entry"] = (common & (close < out["ma200"])
            & (prev_close > close.shift(2)) & (close < prev_close)
            & ((out["upper_repair"] - out["lower_repair"]) >= params.repair_fraction))
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        position = 0
        age = 0
        high_water = low_water = np.nan
        floor = -np.inf
        ceiling = np.inf
        for i in range(len(data)):
            if position:
                age += 1
                if not ready[i] or not np.isfinite(close[i]):
                    position = 0
                else:
                    # Entry fills on this bar when age == 1.
                    high_water = close[i] if age == 1 else max(high_water, close[i])
                    low_water = close[i] if age == 1 else min(low_water, close[i])
                    # The required highest-close floor only ratchets upward.
                    floor = max(floor, high_water - params.trail_atr * atr[i])
                    # Shorts also have a mirrored adverse-move trailing stop.
                    ceiling = min(ceiling, low_water + params.trail_atr * atr[i])
                    if close[i] < floor or (position < 0 and close[i] > ceiling) or age >= 2:
                        position = 0
                # An exit bar cannot immediately open another trade.
            elif ready[i] and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                age = 0
                high_water = low_water = np.nan
                floor = -np.inf
                ceiling = np.inf
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeRepairParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 100., 102., 100., 100., 100.]}, index=index)
    ind = pd.DataFrame({
        "atr": 1., "ready": True,
        "long_entry": [False, True, False, False, False, False],
        "short_entry": False
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    ind["short_entry"] = ind["long_entry"]
    ind["long_entry"] = False
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
