from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeContractionParams:
    lookback: int = 3
    compression_ratio: float = 0.90
    body_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[RangeContractionParams]):
    strategy_id = "gen_local-w2_1790975212"

    @classmethod
    def params_type(cls) -> type[RangeContractionParams]:
        return RangeContractionParams

    def warmup_bars(self, params: RangeContractionParams) -> int:
        return max(2, int(params.lookback)) + 1

    def indicators(self, data: pd.DataFrame, params: RangeContractionParams) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not np.isfinite(params.compression_ratio) or not 0 < params.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if not np.isfinite(params.body_threshold) or not 0 <= params.body_threshold < 1:
            raise ValueError("body_threshold must be in [0, 1)")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["previous_range"] = out["range"].shift(1)
        out["reference_range"] = out["range"].shift(1).rolling(int(params.lookback), min_periods=int(params.lookback)).median()
        out["lower_edge_move"] = data["low"].diff()
        out["upper_edge_retreat"] = -data["high"].diff()
        out["migration"] = (out["lower_edge_move"] - out["upper_edge_retreat"]) / out["previous_range"]
        out["body_pressure"] = (data["close"] - data["open"]) / out["range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["compressed"] = (
            (out["range"] < out["previous_range"])
            & (out["range"] <= params.compression_ratio * out["reference_range"])
        )
        valid = out[["range", "previous_range", "reference_range", "migration", "body_pressure"]].notna().all(axis=1)
        long_entry = valid & out["compressed"] & (out["migration"] > 0) & (out["body_pressure"] > params.body_threshold)
        short_entry = valid & out["compressed"] & (out["migration"] < 0) & (out["body_pressure"] < -params.body_threshold)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeContractionParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Exit on opposite joint agreement; re-entry needs a later bar.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeContractionParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, 0, -1, -1, 0, 1, 1]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
