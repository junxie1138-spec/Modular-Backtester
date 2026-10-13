from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoilParams:
    memory_bars: int = 8
    drift_bars: int = 6
    min_displacement: float = 0.65

    def __post_init__(self):
        if self.memory_bars < 2 or self.drift_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not np.isfinite(self.min_displacement) or self.min_displacement <= 0:
            raise ValueError("min_displacement must be finite and positive.")


class GeneratedStrategy(BaseStrategy[RangeRecoilParams]):
    strategy_id = "gen_local-w2_1791229702"

    @classmethod
    def params_type(cls):
        return RangeRecoilParams

    @staticmethod
    def warmup_bars(params: RangeRecoilParams) -> int:
        return int(max(params.memory_bars + 3, params.drift_bars + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeRecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["relative_range"] = out["range"] / out["midpoint"].where(out["midpoint"] > 0)
        out["range_change"] = out["relative_range"].diff()
        out["prior_range_change"] = out["range_change"].shift(1)
        out["range_product"] = out["range_change"] * out["prior_range_change"]
        out["memory_numerator"] = out["range_product"].rolling(params.memory_bars).sum()
        out["memory_denominator"] = out["range_product"].abs().rolling(params.memory_bars).sum()
        out["range_memory"] = out["memory_numerator"] / out["memory_denominator"].replace(0, np.nan)
        out["drift_scale"] = out["range"].rolling(params.drift_bars).mean()
        out["midpoint_drift"] = (out["midpoint"] - out["midpoint"].shift(params.drift_bars)) / out["drift_scale"]
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out["range_memory"].notna() & out["midpoint_drift"].abs().ge(params.min_displacement)
        direction = np.sign(out["midpoint_drift"].fillna(0))
        out["entry_direction"] = np.where(valid, np.where(out["range_memory"] > 0, direction, -direction), 0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRecoilParams,
    ) -> SignalFrame:
        raw = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        previous = 0
        for i, candidate in enumerate(raw):
            # Opposite evidence exits; two consecutive bars confirm every entry.
            if position != 0 and candidate == -position:
                position = 0
            elif position == 0 and candidate != 0 and candidate == previous:
                position = int(candidate)
            positions[i] = position
            previous = int(candidate)
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeRecoilParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    entries = pd.DataFrame({"entry_direction": [1, 1, 0, 0, -1, -1, 0, 1, 1, 0, -1, -1]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert (flat["entry_direction"] == 0).all()
    assert strategy.warmup_bars(params) >= params.memory_bars + 2


if __name__ == "__main__":
    _self_check()
