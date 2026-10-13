from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMidpointParams:
    range_bars: int = 20
    min_gap_range: float = 0.04
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapMidpointParams]):
    strategy_id = "gen_local-w2_1790980516"

    @classmethod
    def params_type(cls):
        return GapMidpointParams

    def warmup_bars(self, params: GapMidpointParams) -> int:
        if params.range_bars < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError("range_bars must be >= 2 and hold_bars must be 3-5")
        if not np.isfinite(params.min_gap_range) or params.min_gap_range < 0:
            raise ValueError("min_gap_range must be finite and nonnegative")
        return int(params.range_bars + 2)

    def indicators(self, data: pd.DataFrame, params: GapMidpointParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["range_width"] = (out["range_high"] - out["range_low"]).where(lambda s: s > 0)
        out["midpoint"] = (out["range_high"] + out["range_low"]) / 2.0
        out["gap_fraction"] = ((data["open"] - out["prior_close"]) / out["range_width"]).replace([np.inf, -np.inf], np.nan)
        out["first_confirmation"] = (
            (out["prior_close"] <= out["midpoint"])
            & (data["open"] > out["midpoint"])
            & (out["gap_fraction"] >= params.min_gap_range)
            & (data["close"] >= data["open"])
        ).fillna(False)
        out["confirmed"] = (
            out["first_confirmation"].shift(1, fill_value=False)
            & (data["low"] >= out["midpoint"].shift(1))
            & (data["close"] >= data["close"].shift(1))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMidpointParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        candidates = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
            elif candidate:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapMidpointParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"confirmed": True}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["confirmed"].any()
