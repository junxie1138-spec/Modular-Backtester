from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    peak_bars: int = 63
    percentile_bars: int = 126
    correlation_bars: int = 20
    depth_quantile: float = 0.8
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.peak_bars, self.percentile_bars, self.correlation_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars.")
        if self.hold_bars < 1 or not 0.0 < self.depth_quantile < 1.0:
            raise ValueError("Invalid holding period or percentile.")


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791254553"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return int(params.peak_bars + max(params.percentile_bars, params.correlation_bars + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = 1.0 - close / peak
        impulse = depth.diff()
        lagged_impulse = impulse.shift(1)
        correlation = impulse.rolling(
            params.correlation_bars, min_periods=params.correlation_bars
        ).corr(lagged_impulse).clip(-1.0, 1.0)
        threshold = depth.shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.depth_quantile)
        return pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "impulse": impulse,
            "lagged_impulse": lagged_impulse,
            "correlation": correlation,
            "depth_threshold": threshold,
            "predicted_deepening": correlation * impulse,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators[[
            "depth", "depth_threshold", "correlation", "predicted_deepening"
        ]]).all(axis=1)
        extreme = valid & (indicators["depth"] > 0.0) & (
            indicators["depth"] > indicators["depth_threshold"]
        )
        forecast = indicators["predicted_deepening"]
        entries = np.select(
            [extreme & (forecast < 0.0), extreme & (forecast > 0.0)],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = ShockwaveParams()
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ind = pd.DataFrame({
        "depth": 0.2, "depth_threshold": 0.1,
        "correlation": 0.5, "predicted_deepening": -0.01,
    }, index=index)
    ind.loc[index[3]:, "predicted_deepening"] = 0.01
    signal = GeneratedStrategy.generate_signals(data, ind, None, p).data["signal"]
    assert signal.iloc[0] == 0
    assert (signal.iloc[1:19] == 1).all() and signal.iloc[19] == 0
    assert (signal.iloc[20:38] == -1).all() and signal.iloc[38] == 0
    actual = GeneratedStrategy.indicators(data, p)
    assert GeneratedStrategy.generate_signals(data, actual, None, p).data["signal"].eq(0).all()
