from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeChangeParams:
    range_ratio: float = 1.30
    efficiency_min: float = 0.55


class GeneratedStrategy(BaseStrategy[RangeChangeParams]):
    strategy_id = "gen_local_1791274809"

    @classmethod
    def params_type(cls) -> type[RangeChangeParams]:
        return RangeChangeParams

    def warmup_bars(self, params: RangeChangeParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: RangeChangeParams) -> pd.DataFrame:
        if not np.isfinite(params.range_ratio) or params.range_ratio <= 1.0:
            raise ValueError("range_ratio must exceed one")
        if not np.isfinite(params.efficiency_min) or not 0.0 <= params.efficiency_min <= 1.0:
            raise ValueError("efficiency_min must lie between zero and one")
        out = pd.DataFrame(index=data.index)
        out["range"] = ((data["high"] - data["low"]) / data["close"].where(data["close"] > 0)).where(data["high"] >= data["low"])
        out["recent_range"] = out["range"].rolling(3, min_periods=3).mean()
        out["baseline_range"] = out["range"].shift(3).rolling(5, min_periods=5).mean()
        out["range_ratio"] = out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_step"] = out["midpoint"].diff()
        out["migration"] = out["midpoint_step"].rolling(3, min_periods=3).sum()
        out["travel"] = out["midpoint_step"].abs().rolling(3, min_periods=3).sum()
        out["efficiency"] = out["migration"].abs() / out["travel"].where(out["travel"] > 0)
        valid = np.isfinite(out["range_ratio"]) & np.isfinite(out["efficiency"]) & np.isfinite(out["migration"])
        qualified = valid & (out["efficiency"] >= params.efficiency_min)
        direction = np.sign(out["migration"]).fillna(0).astype(int)
        out["entry_direction"] = 0
        expansion = qualified & (out["range_ratio"] >= params.range_ratio)
        contraction = qualified & (out["range_ratio"] <= 1.0 / params.range_ratio)
        out.loc[expansion, "entry_direction"] = direction.loc[expansion]
        out.loc[contraction, "entry_direction"] = -direction.loc[contraction]
        out["entry_direction"] = out["entry_direction"].astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeChangeParams) -> SignalFrame:
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeChangeParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    actual = GeneratedStrategy.indicators(None, data, params)
    assert actual["entry_direction"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(None, params) <= 10
    evidence = pd.DataFrame({"entry_direction": [0, 1, 0, -1, -1, 0, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
