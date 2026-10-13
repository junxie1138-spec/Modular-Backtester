from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    recovery_window: int = 3
    percentile_window: int = 63
    acceleration_quantile: float = 0.8
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790889171"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.recovery_window + 1, params.percentile_window + 3))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.recovery_window < 1 or params.percentile_window < 2:
            raise ValueError("Lookback windows must be positive; percentile_window must be at least two.")
        if not 0.5 < params.acceleration_quantile < 1.0:
            raise ValueError("acceleration_quantile must be between 0.5 and 1.")
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        result = pd.DataFrame(index=data.index)
        result["daily_return"] = close.pct_change(fill_method=None)
        result["acceleration"] = result["daily_return"].diff()
        result["recovery_return"] = close.pct_change(periods=params.recovery_window, fill_method=None)
        result["acceleration_threshold"] = (
            result["acceleration"].abs()
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.acceleration_quantile)
            .shift(1)
        )
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be finite and positive.")
        ready = indicators.notna().all(axis=1)
        entry_condition = (
            ready
            & indicators["daily_return"].gt(0)
            & indicators["recovery_return"].lt(0)
            & indicators["acceleration"].gt(indicators["acceleration_threshold"])
        )
        df = pd.DataFrame(index=data.index)
        # Remain long only while the entry condition holds; its reversal exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({
        "daily_return": [0.0, 0.02, 0.03, 0.0, 0.04, 0.0, 0.0],
        "acceleration": [0.0, 0.02, 0.03, 0.0, 0.04, 0.0, 0.0],
        "recovery_return": [-0.05] * 7,
        "acceleration_threshold": [np.nan, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RecoveryParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
