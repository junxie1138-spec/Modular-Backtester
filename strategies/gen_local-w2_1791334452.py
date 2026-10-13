from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    range_lookback: int = 20
    expansion_threshold: float = 1.5
    contraction_threshold: float = 0.5
    minimum_gap: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = "gen_local-w2_1791334452"

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return max(200, params.range_lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        if params.range_lookback < 2:
            raise ValueError("range_lookback must be at least 2")
        if not np.isfinite(params.expansion_threshold) or params.expansion_threshold <= 1:
            raise ValueError("expansion_threshold must exceed 1")
        if not np.isfinite(params.contraction_threshold) or not 0 < params.contraction_threshold < 1:
            raise ValueError("contraction_threshold must lie between 0 and 1")
        if not np.isfinite(params.minimum_gap) or params.minimum_gap <= 0:
            raise ValueError("minimum_gap must be positive")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["previous_range"] = out["range"].shift(1)
        out["baseline_range"] = out["range"].rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).median().shift(2)
        out["previous_close"] = data["close"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["prior_expansion"] = out["previous_range"] / out["baseline_range"]
        out["contraction"] = out["range"] / out["previous_range"]
        out["gap_depth"] = (out["previous_close"] - data["open"]) / out["previous_range"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            (data["close"] > out["ma200"])
            & (out["prior_expansion"] >= params.expansion_threshold)
            & (out["contraction"] <= params.contraction_threshold)
            & (out["gap_depth"] >= params.minimum_gap)
            & (data["high"] < out["previous_close"])
            & (data["low"] >= out["previous_low"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCompressionParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        entries = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                else:
                    signal[i] = 1
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    indicators = pd.DataFrame({"setup": True}, index=index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
    ):
        result = strategy.generate_signals(
            data, indicators, None, GapCompressionParams(hold_bars=hold)
        )
        assert result.data["signal"].tolist() == expected
        assert (result.data["size"] > 0).all()
    indicators["setup"] = [True] + [False] * 9
    result = strategy.generate_signals(data, indicators, None, GapCompressionParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    warmup = strategy.indicators(data, GapCompressionParams())
    assert not warmup["setup"].any()


if __name__ == "__main__":
    _self_check()
