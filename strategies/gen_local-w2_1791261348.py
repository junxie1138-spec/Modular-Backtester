from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    slope_window: int = 12
    innovation_window: int = 3
    drawdown_window: int = 20
    volatility_window: int = 20
    innovation_threshold: float = 0.15
    minimum_drawdown: float = 0.01


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791261348"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.slope_window + params.innovation_window,
            params.drawdown_window,
            params.volatility_window + 1,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.slope_window, params.innovation_window,
               params.drawdown_window, params.volatility_window) < 2:
            raise ValueError("All windows must be at least two bars")
        if params.innovation_threshold < 0 or params.minimum_drawdown <= 0:
            raise ValueError("Invalid recovery thresholds")

        close = data["close"].astype(float).where(data["close"] > 0)
        log_close = np.log(close)
        returns = log_close.diff()
        n = params.slope_window
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        slope = sum(
            (center - lag) * log_close.shift(lag)
            for lag in range(n)
        ) / denominator
        expected_return = slope.shift(1)
        innovation = returns - expected_return
        smoothed_innovation = innovation.rolling(
            params.innovation_window,
            min_periods=params.innovation_window,
        ).mean()
        volatility = returns.rolling(
            params.volatility_window,
            min_periods=params.volatility_window,
        ).std(ddof=0).replace(0.0, np.nan)
        innovation_score = smoothed_innovation / volatility
        peak = close.rolling(
            params.drawdown_window,
            min_periods=params.drawdown_window,
        ).max()
        drawdown = close / peak - 1.0
        entry_condition = (
            (drawdown <= -params.minimum_drawdown)
            & (slope < 0.0)
            & (innovation_score > params.innovation_threshold)
        ).fillna(False)

        return pd.DataFrame({
            "log_close": log_close,
            "log_return": returns,
            "latent_slope": slope,
            "expected_return": expected_return,
            "innovation": innovation,
            "smoothed_innovation": smoothed_innovation,
            "volatility": volatility,
            "innovation_score": innovation_score,
            "rolling_peak": peak,
            "drawdown": drawdown,
            "entry_condition": entry_condition,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        condition = indicators["entry_condition"].fillna(False).astype(bool)
        # Two consecutive qualifying closes enter; the first false close exits.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry_condition": [False, True, True, True, False, True, True, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert (result["size"] > 0.0).all()
    flat = strategy.indicators(data, RecoveryParams())
    assert not flat["entry_condition"].any()
