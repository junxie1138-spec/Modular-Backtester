from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    entry_z: float = 0.8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791295437"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.lookback) + 4

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if isinstance(params.lookback, bool) or not isinstance(params.lookback, int) or params.lookback < 4:
            raise ValueError("lookback must be an integer >= 4")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        close = data["close"].astype(float)
        ma = close.rolling(params.lookback, min_periods=params.lookback).mean()
        scale = close.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        z = ((close - ma) / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({"ma": ma, "scale": scale, "z": z, "ma_drift": ma - ma.shift(4)}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        z = indicators["z"]
        previous = z.shift(1)
        older = z.shift(2)
        oldest = z.shift(3)
        drift = indicators["ma_drift"]
        valid = pd.concat([z, previous, older, oldest, drift], axis=1).notna().all(axis=1)
        long_entry = valid & (drift > 0) & (previous >= params.entry_z) & (oldest > older) & (older > previous) & (z > previous)
        short_entry = valid & (drift < 0) & (previous <= -params.entry_z) & (oldest < older) & (older < previous) & (z < previous)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                else:
                    signal[bar] = position
                continue
            if entries[bar] != 0:
                position = int(entries[bar])
                exit_bar = bar + 4
                signal[bar] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    for direction in (1, -1):
        indicators = pd.DataFrame({"z": np.array([2.0, 2.0, 2.0, 1.8, 1.6, 1.7, 2.0, 2.1, 2.2, 2.3, 2.4]) * direction, "ma_drift": float(direction)}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, StrategyParams()).data
        assert result["signal"].tolist() == [0] * 6 + [direction] * 4 + [0]
        assert (result["size"] > 0).all()
