from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_threshold: float = 0.80
    trail_atr: float = 0.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791286463"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.rank_threshold < 1.0):
            raise ValueError("rank_threshold must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda x: x > 0.0)
        opening = data["open"].where(lambda x: x > 0.0)
        out["overnight"] = (opening / previous - 1.0).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"] / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        out["dislocation"] = out["overnight"] - out["intraday"]
        out["rank"] = out["dislocation"].rolling(63, min_periods=63).rank(pct=True)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["thursday"] = pd.Series(data.index.dayofweek == 3, index=data.index)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = (
            indicators["thursday"]
            & indicators["rank"].ge(params.rank_threshold)
            & indicators["overnight"].gt(0.0)
            & indicators["intraday"].lt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and close[i] > 0.0
                    and np.isfinite(atr[i]) and atr[i] > 0.0):
                active = False
                age = 0
                high_water = stop = -np.inf
                continue
            if active:
                # Entry fills at this bar's open; only subsequent closes count.
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] < stop or age >= 2:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                else:
                    signal[i] = 1
            elif entries[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-01-04", periods=4, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 98.0,
                         "close": [100.0, 101.0, 99.0, 100.0],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0, "thursday": [True, False, False, False],
                               "rank": 0.95, "overnight": 0.01,
                               "intraday": -0.01}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    assert result["size"].gt(0.0).all()
