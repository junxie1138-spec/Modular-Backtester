from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRelocationParams:
    range_lookback: int = 20
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRelocationParams]):
    strategy_id = "gen_local_1790875321"

    @classmethod
    def params_type(cls) -> type[GapRelocationParams]:
        return GapRelocationParams

    def warmup_bars(self, params: GapRelocationParams) -> int:
        return params.range_lookback + 2

    def indicators(self, data: pd.DataFrame, params: GapRelocationParams) -> pd.DataFrame:
        if params.range_lookback < 2 or params.hold_bars < 1:
            raise ValueError("range_lookback must be >= 2 and hold_bars must be >= 1")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["baseline"] = out["range"].rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).median().shift(1)
        out["gap"] = data["open"] - data["high"].shift(1)
        out["gap_floor"] = data["low"].shift(1)
        out["gap_midpoint"] = (data["high"].shift(1) + data["low"].shift(1)) / 2.0
        out["relocation"] = (
            out["gap"].ge(0.25 * out["baseline"])
            & out["range"].le(0.8 * out["baseline"])
        ).fillna(False)
        out["entry"] = (
            out["relocation"].shift(1, fill_value=False)
            & out["range"].le(0.75 * out["range"].shift(1))
            & data["low"].ge(out["gap_floor"])
            & data["close"].ge(out["gap_midpoint"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRelocationParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be >= 1")
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapRelocationParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=60),
    )
    data.iloc[25] = [102.0, 103.0, 102.0, 102.7, 1000000.0]
    data.iloc[26] = [102.6, 102.95, 102.3, 102.8, 1000000.0]
    derived = strategy.indicators(data, params)
    assert np.flatnonzero(derived["entry"].to_numpy()).tolist() == [26]
    result = strategy.generate_signals(data, derived, None, params).data
    assert np.flatnonzero(result["signal"].to_numpy()).tolist() == list(range(27, 45))
    assert result["size"].gt(0).all()
    derived["entry"] = np.arange(len(data)) >= 26
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[27:45].eq(1).all()
    assert result["signal"].iloc[45] == 0
    assert result["signal"].iloc[46] == 1
