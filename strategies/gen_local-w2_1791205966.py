from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    lookback: int = 20
    gap_capacity: float = 1.25

    def __post_init__(self):
        if self.lookback < 2 or not np.isfinite(self.gap_capacity) or self.gap_capacity <= 0:
            raise ValueError("lookback must be >= 2 and gap_capacity must be finite and positive")


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local-w2_1791205966"

    @classmethod
    def params_type(cls):
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["downside_gap"] = (out["previous_close"] - data["open"]) / out["previous_close"]
        out["upward_travel"] = (data["high"] - data["open"]).clip(lower=0) / out["previous_close"]
        out["capacity"] = out["upward_travel"].rolling(
            params.lookback, min_periods=params.lookback
        ).median().shift(1)
        out["repair"] = (data["close"] - data["open"]) / out["previous_close"]
        out["residual"] = (out["previous_close"] - data["close"]) / out["previous_close"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCapacityParams,
    ) -> SignalFrame:
        entry = (
            indicators["capacity"].gt(0)
            & indicators["downside_gap"].gt(params.gap_capacity * indicators["capacity"])
            & indicators["repair"].gt(0)
            & indicators["residual"].gt(0)
        )
        df = pd.DataFrame(index=data.index)
        # Stay long only while the entry condition holds; its reversal is the sole exit.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 97.0, 98.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 99.0, 99.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 96.0, 97.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0, 98.0, 98.0, 100.0],
            "volume": [1000.0] * 7,
        },
        index=pd.date_range("2020-01-01", periods=7),
    )
    strategy = GeneratedStrategy()
    params = GapCapacityParams(lookback=3)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), context, params)
    assert result.data["signal"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert result.data["size"].gt(0).all()
