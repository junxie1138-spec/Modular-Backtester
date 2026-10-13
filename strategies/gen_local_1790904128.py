from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    window: int = 20
    acceleration_floor: float = 0.5

    def __post_init__(self):
        if not isinstance(self.window, int) or self.window < 2:
            raise ValueError("window must be an integer >= 2")
        if not np.isfinite(self.acceleration_floor) or self.acceleration_floor < 0:
            raise ValueError("acceleration_floor must be finite and nonnegative")


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = "gen_local_1790904128"

    @classmethod
    def params_type(cls):
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return params.window + 1

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap_roc"] = data["open"] / out["previous_close"] - 1.0
        out["intraday_roc"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["recovery_roc"] = out["intraday_roc"] - out["gap_roc"]
        out["recovery_acceleration"] = out["recovery_roc"].diff()
        out["gap_scale"] = out["gap_roc"].abs().rolling(
            params.window, min_periods=params.window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRecoveryParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry = (
            ready
            & (indicators["gap_roc"] < 0.0)
            & (indicators["intraday_roc"] > 0.0)
            & (
                indicators["recovery_acceleration"]
                > params.acceleration_floor * indicators["gap_scale"]
            )
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The position exits exactly when the same entry condition flips false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapRecoveryParams(window=2)
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.warmup_bars(params) == 3
    assert actual["gap_scale"].iloc[:2].isna().all()
    assert strategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()
    probe = pd.DataFrame(
        {"previous_close": 100.0, "gap_roc": -0.01,
         "intraday_roc": 0.01, "recovery_roc": 0.02,
         "recovery_acceleration": [1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
         "gap_scale": 0.01}, index=index
    )
    result = strategy.generate_signals(data, probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
