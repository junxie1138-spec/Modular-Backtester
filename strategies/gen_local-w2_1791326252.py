from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCapacityParams:
    capacity_window: int = 40
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeCapacityParams]):
    strategy_id = "gen_local-w2_1791326252"

    @classmethod
    def params_type(cls) -> type[RangeCapacityParams]:
        return RangeCapacityParams

    def warmup_bars(self, params: RangeCapacityParams) -> int:
        if params.capacity_window < 2 or params.hold_bars < 1:
            raise ValueError("capacity_window must be >= 2 and hold_bars must be >= 1")
        return int(params.capacity_window + 5)

    def indicators(self, data: pd.DataFrame, params: RangeCapacityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["capacity"] = out["range"].rolling(params.capacity_window, min_periods=params.capacity_window).mean().shift(1)
        out["excess"] = (out["range"] / out["capacity"].replace(0, np.nan) - 1.0).clip(lower=0).replace([np.inf, -np.inf], np.nan)
        out["overflow"] = out["excess"].rolling(5, min_periods=5).sum()
        out["intersection"] = (pd.concat([data["high"], data["high"].shift(1)], axis=1).min(axis=1, skipna=False) - pd.concat([data["low"], data["low"].shift(1)], axis=1).max(axis=1, skipna=False)).clip(lower=0)
        out["overlap_capacity"] = pd.concat([out["range"], out["range"].shift(1)], axis=1).min(axis=1, skipna=False)
        out["overlap"] = (out["intersection"] / out["overlap_capacity"].replace(0, np.nan)).clip(0, 1).replace([np.inf, -np.inf], np.nan)
        out["congestion"] = out["overlap"].rolling(5, min_periods=5).mean()
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["migration"] = out["midpoint"].diff().rolling(5, min_periods=5).sum()
        out["regime_polarity"] = pd.Series(np.where(out["congestion"] >= 0.6, -1.0, 1.0), index=data.index).where(out["congestion"].notna())
        out["direction"] = np.sign(out["migration"]) * out["regime_polarity"]
        out["overflow_event"] = ((out["overflow"] >= 2.0) & (out["overflow"].shift(1) < 2.0)).astype(int)
        ready = out[["overflow", "congestion", "migration", "direction"]].notna().all(axis=1)
        out["candidate"] = out["direction"].where(ready & out["overflow_event"].eq(1), 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeCapacityParams) -> SignalFrame:
        self.warmup_bars(params)
        candidates = indicators["candidate"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeCapacityParams()
    index = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = strategy.indicators(data, params)
    assert indicators["candidate"].eq(0).all()
    indicators.loc[index[[2, 4, 20, 22]], "candidate"] = [1, -1, 1, -1]
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(100, dtype=int)
    expected[3:21] = 1
    expected[23:41] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
