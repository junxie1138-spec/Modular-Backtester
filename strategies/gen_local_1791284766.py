from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.55
    min_displacement: float = 0.35
    max_recoil: float = 0.45


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791284766"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if not (np.isfinite(params.min_displacement) and params.min_displacement > 0.0):
            raise ValueError("min_displacement must be finite and positive")
        if not (0.0 < params.max_recoil < 1.0):
            raise ValueError("max_recoil must be between zero and one")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        weights = [1.0]
        for k in range(1, 5):
            weights.append(-weights[-1] * (params.fractional_order - k + 1) / k)
        fractional_close = close.copy()
        for k in range(1, 5):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        # ponytail: five taps cap memory; longer kernels require relaxing the warmup constraint.
        displacement = fractional_close - sum(weights) * close.shift(4)
        daily_range = (data["high"] - data["low"]).astype(float)
        daily_range = daily_range.replace([np.inf, -np.inf], np.nan).where(daily_range > 0.0)
        range_scale = daily_range.rolling(5, min_periods=5).mean()
        change = close.diff()
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "displacement": displacement,
            "range_scale": range_scale,
            "change": change,
            "normalized_displacement": displacement / range_scale,
            "recoil_fraction": -change / displacement.where(displacement > 0.0),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        # A retained bullish displacement with a small negative close change is the entry condition.
        # Every false condition closes an existing position; there are no other exits.
        entry = (
            (indicators["normalized_displacement"] > params.min_displacement)
            & (indicators["change"] < 0.0)
            & (indicators["recoil_fraction"] > 0.0)
            & (indicators["recoil_fraction"] <= params.max_recoil)
        ).fillna(False)
        df = pd.DataFrame({"signal": entry.astype(int), "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 103.8, 103.7, 105.0, 104.9])
    sample = pd.DataFrame({
        "open": prices, "high": prices + 0.5, "low": prices - 0.5,
        "close": prices, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=len(prices)))
    params = Params()
    features = GeneratedStrategy.indicators(sample, params)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    result = GeneratedStrategy.generate_signals(sample, features, ctx, params).data
    assert GeneratedStrategy.warmup_bars(params) <= 10
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert result["size"].gt(0.0).all()
    assert features.iloc[:4]["displacement"].isna().all()
