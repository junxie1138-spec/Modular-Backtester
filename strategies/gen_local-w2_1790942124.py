from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReleaseParams:
    baseline_bars: int = 5
    compression: float = 0.8
    expansion: float = 1.5
    close_edge: float = 0.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeReleaseParams]):
    strategy_id = "gen_local-w2_1790942124"

    @classmethod
    def params_type(cls):
        return RangeReleaseParams

    def warmup_bars(self, params: RangeReleaseParams) -> int:
        if not isinstance(params.baseline_bars, int) or not 2 <= params.baseline_bars <= 8:
            raise ValueError("baseline_bars must be an integer between 2 and 8")
        if not isinstance(params.hold_bars, int) or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer between 3 and 5")
        if not 0.0 < params.compression < 1.0:
            raise ValueError("compression must be between 0 and 1")
        if not np.isfinite(params.expansion) or params.expansion <= 1.0:
            raise ValueError("expansion must be finite and greater than 1")
        if not 0.5 < params.close_edge < 1.0:
            raise ValueError("close_edge must be between 0.5 and 1")
        return params.baseline_bars + 2

    def indicators(self, data: pd.DataFrame, params: RangeReleaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["prior_range"] = out["range"].shift(1)
        out["baseline_range"] = out["range"].shift(2).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        out["compression_ratio"] = out["prior_range"] / out["baseline_range"]
        out["expansion_ratio"] = out["range"] / out["prior_range"]
        out["upper_extension"] = data["high"] - data["high"].shift(1)
        out["lower_extension"] = data["low"].shift(1) - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / out["range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        release = (
            (out["compression_ratio"] <= params.compression)
            & (out["expansion_ratio"] >= params.expansion)
            & out["close_location"].between(0.0, 1.0)
        )
        upward = (
            release
            & (out["upper_extension"] > 0)
            & (out["lower_extension"] <= 0)
            & (out["close_location"] >= params.close_edge)
        )
        downward = (
            release
            & (out["lower_extension"] > 0)
            & (out["upper_extension"] <= 0)
            & (out["close_location"] <= 1.0 - params.close_edge)
        )
        out["entry_signal"] = np.select([upward, downward], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeReleaseParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry_signal"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0:
                if i - entry_bar == params.hold_bars:
                    position = 0
                    # Reserve the expiry bar for an explicit flat target.
                signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeReleaseParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame(
        {"entry_signal": [1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    flat = strategy.indicators(data, params)
    assert flat["entry_signal"].eq(0).all()
    assert strategy.warmup_bars(params) <= 10
