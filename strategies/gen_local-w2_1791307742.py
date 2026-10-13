from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    recovery_window: int = 20
    roc_window: int = 10
    acceleration_lag: int = 5
    smoothing_window: int = 15
    floor_window: int = 15
    floor_lag: int = 5
    min_drawdown: float = 0.03


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791307742"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(
            params.peak_window + params.recovery_window - 1,
            params.roc_window + params.acceleration_lag + params.smoothing_window,
            params.floor_window + params.floor_lag,
        )

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        windows = (
            params.peak_window, params.recovery_window, params.roc_window,
            params.acceleration_lag, params.smoothing_window,
            params.floor_window, params.floor_lag,
        )
        if any(w < 1 for w in windows) or not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("Windows must be positive and min_drawdown must be between zero and one.")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["recent_depth"] = out["drawdown"].rolling(
            params.recovery_window, min_periods=params.recovery_window
        ).max()
        out["roc"] = close / close.shift(params.roc_window) - 1.0
        out["roc_acceleration"] = out["roc"].diff(params.acceleration_lag)
        out["acceleration"] = out["roc_acceleration"].rolling(
            params.smoothing_window, min_periods=params.smoothing_window
        ).mean()
        out["floor"] = data["low"].where(data["low"] > 0).rolling(
            params.floor_window, min_periods=params.floor_window
        ).min()
        out["floor_lift"] = out["floor"] / out["floor"].shift(params.floor_lag) - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        columns = ["recent_depth", "acceleration", "floor_lift"]
        valid = indicators[columns].notna().all(axis=1)
        recovery_accelerates = (
            (indicators["recent_depth"] >= params.min_drawdown)
            & (indicators["acceleration"] > 0.0)
        )
        downside_frontier_rises = indicators["floor_lift"] > 0.0
        entry_condition = valid & recovery_accelerates & downside_frontier_rises
        df = pd.DataFrame(index=data.index)
        # Exposure ends only when the joint entry condition becomes false.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "recent_depth": [np.nan, 0.04, 0.04, 0.04, 0.04, 0.04],
        "acceleration": [np.nan, 0.01, 0.01, -0.01, 0.01, 0.01],
        "floor_lift": [np.nan, 0.01, -0.01, 0.01, 0.01, 0.01],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, RecoveryParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 0, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
