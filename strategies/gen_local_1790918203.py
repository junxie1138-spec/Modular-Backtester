from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MotifParams:
    repertoire_window: int = 12
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[MotifParams]):
    strategy_id = "gen_local_1790918203"

    @classmethod
    def params_type(cls) -> type[MotifParams]:
        return MotifParams

    @staticmethod
    def warmup_bars(params: MotifParams) -> int:
        return max(3, int(params.repertoire_window)) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: MotifParams) -> pd.DataFrame:
        w = max(3, int(params.repertoire_window))
        out = pd.DataFrame(index=data.index)
        out["span"] = data["high"] - data["low"]
        out["location"] = ((data["close"] - data["low"]) / out["span"].where(out["span"] > 0)).clip(0, 1)
        out["location_lag1"] = out["location"].shift(1)
        out["location_lag2"] = out["location"].shift(2)
        a = out["location_lag2"]
        b = out["location_lag1"]
        c = out["location"]
        out["pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan,
        )
        columns = []
        out["novel_pattern"] = False
        for code in range(6):
            name = f"prior_count_{code}"
            columns.append(name)
            out[name] = out["pattern"].eq(code).astype(float).rolling(w, min_periods=w).sum().shift(1)
            out["novel_pattern"] |= out["pattern"].eq(code) & out[name].eq(0)
        out["complexity"] = out[columns].gt(0).sum(axis=1).where(out[columns].notna().all(axis=1))
        out["valid_patterns"] = out["pattern"].rolling(w + 1, min_periods=w + 1).count().eq(w + 1)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["span"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1,
        ).max(axis=1)
        out["recent_range"] = out["true_range"].rolling(3, min_periods=3).mean().shift(1)
        out["baseline_range"] = out["true_range"].rolling(w, min_periods=w).mean().shift(1)
        out["compression"] = out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0)
        out["upper_boundary"] = data["high"].rolling(3, min_periods=3).max().shift(1)
        out["lower_boundary"] = data["low"].rolling(3, min_periods=3).min().shift(1)
        eligible = out["valid_patterns"] & out["complexity"].le(3) & out["novel_pattern"] & out["compression"].lt(0.75)
        out["entry"] = np.select(
            [eligible & data["close"].gt(out["upper_boundary"]),
             eligible & data["close"].lt(out["lower_boundary"])],
            [1, -1], default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MotifParams) -> SignalFrame:
        holding = int(params.holding_bars)
        if holding < 3 or holding > 5:
            raise ValueError("holding_bars must be between 3 and 5")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        for i, entry in enumerate(entries):
            if position:
                age += 1
                if age == holding:
                    position = 0
                    age = 0
            elif entry:
                position = int(entry)
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    sample = pd.DataFrame(index=index)
    entries = pd.DataFrame({"entry": [1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(sample, entries, None, MotifParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 1]
    assert result["size"].gt(0).all()
