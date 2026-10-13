from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    lookback: int = 20
    price_threshold: float = 0.25
    volume_threshold: float = 0.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1790976073"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        if params.lookback < 3 or params.hold_bars not in (1, 2):
            raise ValueError("lookback must be >= 3 and hold_bars must be 1 or 2")
        if not np.isfinite([params.price_threshold, params.volume_threshold]).all():
            raise ValueError("Thresholds must be finite")
        if min(params.price_threshold, params.volume_threshold) < 0:
            raise ValueError("Thresholds must be nonnegative")
        return params.lookback + 3

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["price_roc"] = data["close"].div(data["close"].shift(1).where(lambda x: x > 0)).sub(1)
        out["volume_roc"] = data["volume"].div(data["volume"].shift(1).where(lambda x: x > 0)).sub(1)
        out["price_acceleration"] = out["price_roc"].diff()
        out["volume_acceleration"] = out["volume_roc"].diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        for primitive in ("price", "volume"):
            acceleration = out[primitive + "_acceleration"]
            scale = acceleration.rolling(params.lookback, min_periods=params.lookback).std()
            out[primitive + "_scale"] = scale
            out[primitive + "_score"] = acceleration.div(scale.where(scale > 0))
        out["acceleration_persistence"] = out["price_acceleration"].rolling(
            params.lookback, min_periods=params.lookback
        ).corr(out["price_acceleration"].shift(1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            (indicators["price_score"] > params.price_threshold)
            & (indicators["acceleration_persistence"] > 0)
            & (indicators["volume_score"] > params.volume_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(eligible):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif entry:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AccelerationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    agreeing = pd.DataFrame(
        {"price_score": 1.0, "volume_score": 1.0, "acceleration_persistence": 0.5},
        index=index,
    )
    result = strategy.generate_signals(data, agreeing, ctx, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    disagreeing = agreeing.copy()
    disagreeing["volume_score"] = -1.0
    assert strategy.generate_signals(data, disagreeing, ctx, params).data["signal"].eq(0).all()
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
