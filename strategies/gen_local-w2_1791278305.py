from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownOccupancyParams:
    peak_window: int = 63
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[DrawdownOccupancyParams]):
    strategy_id = "gen_local-w2_1791278305"

    @classmethod
    def params_type(cls) -> type[DrawdownOccupancyParams]:
        return DrawdownOccupancyParams

    def warmup_bars(self, params: DrawdownOccupancyParams) -> int:
        return max(2, int(params.peak_window)) + 29

    def indicators(self, data: pd.DataFrame, params: DrawdownOccupancyParams) -> pd.DataFrame:
        window = max(2, int(params.peak_window))
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(window, min_periods=window).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].replace(0.0, np.nan)).clip(lower=0.0)
        out["depth_envelope"] = out["depth"].rolling(20, min_periods=20).max()
        valid = out["depth"].notna() & out["depth_envelope"].notna()
        out["deep"] = ((out["depth"] >= 0.5 * out["depth_envelope"]) & (out["depth_envelope"] > 0.0)).astype(float).where(valid)
        out["occupancy"] = out["deep"].rolling(10, min_periods=10).mean()
        out["previous_occupancy"] = out["occupancy"].shift(1)
        ready = out["occupancy"].notna() & out["previous_occupancy"].notna()
        recruitment = ready & (out["occupancy"] >= 0.6) & (out["previous_occupancy"] < 0.6)
        depletion = ready & (out["occupancy"] <= 0.3) & (out["previous_occupancy"] > 0.3)
        out["entry"] = np.select([depletion, recruitment], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownOccupancyParams) -> SignalFrame:
        hold = max(1, int(params.hold_bars))
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + hold
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, 0, -1] + [0] * 17 + [-1] + [0] * 3}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, DrawdownOccupancyParams()).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 2 + [-1] * 3
    assert (result["size"] > 0.0).all()
