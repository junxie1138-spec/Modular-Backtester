from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    lookback: int = 20
    gap_floor: float = 0.5
    excursion_cost: float = 0.5


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1790963317"

    @classmethod
    def params_type(cls):
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        if params.lookback < 1:
            raise ValueError("lookback must be positive")
        if not np.isfinite(params.gap_floor) or params.gap_floor < 0:
            raise ValueError("gap_floor must be finite and nonnegative")
        if not np.isfinite(params.excursion_cost) or params.excursion_cost < 0:
            raise ValueError("excursion_cost must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_scale"] = (
            out["gap"].abs()
            .rolling(params.lookback, min_periods=params.lookback)
            .mean().shift(1).replace(0.0, np.nan)
        )
        out["gap_debt"] = -out["gap"]
        out["fresh_downside"] = (data["open"] - data["low"]).clip(lower=0.0)
        out["repaired_distance"] = data["close"] - data["open"]
        out["repair_surplus"] = (
            out["repaired_distance"] - out["gap_debt"]
            - params.excursion_cost * out["fresh_downside"]
        )
        out["gap_margin"] = out["gap_debt"] / out["gap_scale"] - params.gap_floor
        out["repair_margin"] = out["repair_surplus"] / out["gap_scale"]
        out["eligibility_margin"] = np.minimum(out["gap_margin"], out["repair_margin"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Enter when the predicate becomes true; exit only when it becomes false.
        # The same zero boundary governs both transitions; neutral margins exit.
        df["signal"] = indicators["eligibility_margin"].fillna(0.0).gt(0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0, 99.8, 100.2, 99.0, 100.8, 100.0, 101.7],
            "high": [100.3, 100.3, 100.4, 101.0, 101.1, 102.0, 102.0],
            "low": [99.8, 99.7, 99.8, 98.9, 100.5, 99.9, 101.4],
            "close": [100.0, 100.0, 100.0, 100.8, 100.8, 101.7, 101.7],
            "volume": [1000.0] * 7,
        },
        index=pd.date_range("2020-01-01", periods=7),
    )
    params = GapRepairParams(lookback=2)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 0, 1]
    assert result.index.equals(data.index)
    assert result["size"].gt(0.0).all()
    for length in range(1, len(data) + 1):
        prefix = data.iloc[:length]
        actual = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        pd.testing.assert_frame_equal(actual, result.iloc[:length])


if __name__ == "__main__":
    _self_check()
