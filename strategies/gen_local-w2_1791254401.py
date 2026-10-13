from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    diff_order: float = 0.45
    frac_window: int = 48
    recovery_window: int = 8
    peak_window: int = 40
    min_drawdown: float = 0.015


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791254401"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window,
                       params.frac_window + params.recovery_window - 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.frac_window, params.recovery_window, params.peak_window) < 2:
            raise ValueError("Window lengths must be at least two.")
        if not 0.0 < params.diff_order < 1.0 or not 0.0 < params.min_drawdown < 1.0:
            raise ValueError("Order and drawdown must lie between zero and one.")
        close = data["close"].astype(float).where(lambda s: np.isfinite(s) & (s > 0))
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.diff_order - k + 1) / k
        # ponytail: fixed-width fractional memory; enlarge frac_window for longer memory.
        fractional_close = close * weights[0]
        for k in range(1, params.frac_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["fractional_close"] = fractional_close
        out["recovery_mean"] = fractional_close.rolling(
            params.recovery_window, min_periods=params.recovery_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        ready = np.isfinite(indicators).all(axis=1) & np.isfinite(data["close"])
        condition = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["drawdown"] >= params.min_drawdown)
            & (indicators["fractional_close"] > indicators["recovery_mean"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintain the long only while the complete entry condition remains true.
        df["signal"] = condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({
        "ma200": 99.0, "peak": 103.0, "drawdown": 0.03,
        "fractional_close": [0.0, 2.0, 3.0, 0.0, np.nan],
        "recovery_mean": 1.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
