from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    z_threshold: float = 1.75
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791307538"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 52

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.z_threshold) or params.z_threshold <= 1.0:
            raise ValueError("z_threshold must be finite and greater than one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["std"] = close.rolling(20, min_periods=20).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["overflow"] = np.sign(out["z"]) * (out["z"].abs() - 1.0).clip(lower=0.0)
        out["overflow_change"] = out["overflow"].diff()
        out["lagged_overflow_change"] = out["overflow_change"].shift(1)
        out["overflow_ac"] = out["overflow_change"].rolling(30, min_periods=30).corr(out["lagged_overflow_change"]).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_z"] = out["z"].shift(1)
        ready = np.isfinite(out[["z", "previous_z", "overflow_ac", "atr"]]).all(axis=1) & out["atr"].gt(0.0)
        exhausted = ready & out["overflow_ac"].lt(-0.15)
        long_entry = exhausted & out["previous_z"].lt(-params.z_threshold) & out["z"].ge(-params.z_threshold) & out["z"].lt(0.0)
        short_entry = exhausted & out["previous_z"].gt(params.z_threshold) & out["z"].le(params.z_threshold) & out["z"].gt(0.0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                position = 0
                age = 0
                high_water = low_water = stop = np.nan
                continue
            if position:
                age += 1
                # The first close after the entry decision is the first in-trade close.
                high_water = close[i] if age == 1 else max(high_water, close[i])
                low_water = close[i] if age == 1 else min(low_water, close[i])
                if position == 1:
                    candidate = high_water - params.trail_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close[i] <= stop
                else:
                    candidate = low_water + params.trail_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close[i] >= stop
                if stopped or age >= 4:
                    position = 0
                    age = 0
                    high_water = low_water = stop = np.nan
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water = low_water = stop = np.nan
            decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": [100.0, 100.0, 103.0, 99.0, 100.0, 100.0, 97.0, 101.0, 100.0]}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    data["close"] = 100.0
    indicators["entry"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
