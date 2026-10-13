from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAccelerationParams:
    range_window: int = 63
    roc_window: int = 5
    acceleration_lag: int = 3
    edge_fraction: float = 0.35
    acceleration_floor: float = 0.002
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeAccelerationParams]):
    strategy_id = "gen_local_1791256582"

    @classmethod
    def params_type(cls) -> type[RangeAccelerationParams]:
        return RangeAccelerationParams

    def warmup_bars(self, params: RangeAccelerationParams) -> int:
        if min(params.range_window, params.roc_window, params.acceleration_lag, params.hold_bars) < 1:
            raise ValueError("Lookbacks and hold_bars must be positive")
        if not 0.0 < params.edge_fraction < 0.5 or not np.isfinite(params.acceleration_floor) or params.acceleration_floor < 0.0:
            raise ValueError("Invalid entry thresholds")
        return int(max(params.range_window + 2, params.roc_window + params.acceleration_lag + 2))

    def indicators(self, data: pd.DataFrame, params: RangeAccelerationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        result["range_low"] = data["low"].rolling(params.range_window, min_periods=params.range_window).min().shift(1)
        result["range_high"] = data["high"].rolling(params.range_window, min_periods=params.range_window).max().shift(1)
        result["range_span"] = result["range_high"] - result["range_low"]
        result["position"] = (data["close"] - result["range_low"]) / result["range_span"].where(result["range_span"] > 0.0)
        result["roc"] = data["close"].pct_change(params.roc_window, fill_method=None)
        result["acceleration"] = result["roc"].diff(params.acceleration_lag)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeAccelerationParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        values = indicators[["position", "roc", "acceleration"]].replace([np.inf, -np.inf], np.nan)
        valid = values.notna().all(axis=1)
        long_setup = valid & values["position"].between(0.0, params.edge_fraction) & (values["roc"] > 0.0) & (values["acceleration"] > params.acceleration_floor)
        short_setup = valid & values["position"].between(1.0 - params.edge_fraction, 1.0) & (values["roc"] < 0.0) & (values["acceleration"] < -params.acceleration_floor)
        long_confirmed = long_setup & long_setup.shift(1, fill_value=False)
        short_confirmed = short_setup & short_setup.shift(1, fill_value=False)
        entries = np.where(long_confirmed, 1, np.where(short_confirmed, -1, 0))
        signals = np.zeros(len(data), dtype=np.int64)
        position = 0
        entry_bar = -1
        for bar, candidate in enumerate(entries):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
                # Opposite setups never change an active position or its expiry.
            elif candidate != 0:
                position = int(candidate)
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeAccelerationParams()
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    values = pd.DataFrame({"position": [0.2] * 2 + [0.8] * 23, "roc": [0.01] * 2 + [-0.01] * 23, "acceleration": [0.01] * 2 + [-0.01] * 23}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, values, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20] == 0
    assert result["signal"].iloc[21:].eq(-1).all()
    assert result["size"].gt(0).all()
