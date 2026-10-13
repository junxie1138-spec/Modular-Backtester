from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    range_window: int = 3
    memory: int = 6
    capacity_fraction: float = 0.5
    entry_threshold: float = 0.1


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local-w2_1791260688"

    @classmethod
    def params_type(cls) -> type[GapCapacityParams]:
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        if not isinstance(params.range_window, int) or not isinstance(params.memory, int):
            raise ValueError("Lookbacks must be integers")
        bars = params.range_window + params.memory
        if params.range_window < 1 or params.memory < 1 or bars > 10:
            raise ValueError("Positive lookbacks must sum to at most 10")
        if not np.isfinite(params.capacity_fraction) or params.capacity_fraction <= 0:
            raise ValueError("capacity_fraction must be finite and positive")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold < 0:
            raise ValueError("entry_threshold must be finite and nonnegative")
        return int(bars)

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["capacity"] = (
            out["range"].shift(1).rolling(params.range_window, min_periods=params.range_window).min()
            * params.capacity_fraction
        )
        out["overflow"] = np.sign(out["gap"]) * (out["gap"].abs() - out["capacity"]).clip(lower=0.0)
        out["overflow_sum"] = out["overflow"].rolling(params.memory, min_periods=params.memory).sum()
        out["capacity_sum"] = out["capacity"].rolling(params.memory, min_periods=params.memory).sum()
        out["pressure"] = (
            out["overflow_sum"] / out["capacity_sum"].where(out["capacity_sum"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCapacityParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The long entry condition reversing to false is the sole exit rule.
        df["signal"] = indicators["pressure"].gt(params.entry_threshold).fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapCapacityParams()
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 9
    assert derived["pressure"].first_valid_index() == index[8]
    assert derived["pressure"].dropna().eq(0.0).all()
    probe = pd.DataFrame({"pressure": [np.nan, 0.2, 0.3, 0.0, -0.2]}, index=index[:5])
    result = strategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc("open")] = 110.0
    pd.testing.assert_frame_equal(strategy.indicators(changed, params).iloc[:-1], derived.iloc[:-1])


if __name__ == "__main__":
    _self_check()
