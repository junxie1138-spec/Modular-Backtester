from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_window: int = 8
    flow_window: int = 8
    gap_threshold: float = 0.25
    flow_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790940177"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.gap_window, params.flow_window) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.gap_window < 2 or params.flow_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0 < params.gap_threshold <= 1 and 0 < params.flow_threshold <= 1):
            raise ValueError("Agreement thresholds must lie in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] / out["previous_close"].where(out["previous_close"] > 0) - 1
        out["gap_vote"] = np.sign(out["gap"])
        out["gap_breadth"] = out["gap_vote"].rolling(params.gap_window).mean()
        out["close_change"] = data["close"].diff()
        out["close_vote"] = np.sign(out["close_change"])
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["signed_volume"] = out["close_vote"] * out["valid_volume"]
        out["volume_sum"] = out["valid_volume"].rolling(params.flow_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.flow_window).sum()
        out["volume_pressure"] = out["signed_volume_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        breadth = indicators["gap_breadth"]
        pressure = indicators["volume_pressure"]
        long_entry = (breadth >= params.gap_threshold) & (pressure >= params.flow_threshold)
        short_entry = (breadth <= -params.gap_threshold) & (pressure <= -params.flow_threshold)
        agreement = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(agreement):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite AND confirmation exits to flat; re-entry needs a later bar.
                position = 0
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    scores = [0.5, 0.0, -0.5, -0.5, np.nan, 0.5, 0.5]
    indicators = pd.DataFrame({"gap_breadth": scores, "volume_pressure": scores}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
