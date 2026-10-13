from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBurdenParams:
    peak_window: int = 84
    recovery_window: int = 18
    burden_ratio: float = 0.65
    minimum_prior_depth: float = 0.01

    def __post_init__(self):
        if self.peak_window < 2 or self.recovery_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not 0.0 < self.burden_ratio < 1.0 or not 0.0 < self.minimum_prior_depth < 1.0:
            raise ValueError("Depth thresholds must lie strictly between zero and one.")


class GeneratedStrategy(BaseStrategy[DrawdownBurdenParams]):
    strategy_id = "gen_local-w2_1791319117"

    @classmethod
    def params_type(cls):
        return DrawdownBurdenParams

    def warmup_bars(self, params: DrawdownBurdenParams) -> int:
        return int(params.peak_window + 2 * params.recovery_window)

    def indicators(self, data: pd.DataFrame, params: DrawdownBurdenParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - close / peak).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
        recent = depth.rolling(params.recovery_window, min_periods=params.recovery_window).mean()
        prior = recent.shift(params.recovery_window)
        return pd.DataFrame({
            "peak": peak,
            "drawdown_depth": depth,
            "recent_burden": recent,
            "prior_burden": prior,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownBurdenParams,
    ) -> SignalFrame:
        recent = indicators["recent_burden"]
        prior = indicators["prior_burden"]
        condition = (
            np.isfinite(recent)
            & np.isfinite(prior)
            & (prior >= params.minimum_prior_depth)
            & (recent < params.burden_ratio * prior)
        ).fillna(False)
        # Two true closes enter; the first false close reverses to flat.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame({"signal": confirmed.astype(int), "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({
        "recent_burden": [np.nan, 0.005, 0.005, 0.02, 0.005, 0.005, 0.005, 0.02],
        "prior_burden": [np.nan, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
    }, index=index)
    strategy = GeneratedStrategy()
    params = DrawdownBurdenParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 1, 1]
    assert result.index.equals(data.index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
