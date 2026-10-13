from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 8
    vol_window: int = 5
    min_depth: float = 0.8
    max_depth: float = 2.5
    target_daily_vol: float = 0.006


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791239059"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    @staticmethod
    def warmup_bars(params: DrawdownParams) -> int:
        if not 2 <= params.peak_window <= 8:
            raise ValueError("peak_window must be between 2 and 8")
        if not 2 <= params.vol_window <= 9:
            raise ValueError("vol_window must be between 2 and 9")
        values = (params.min_depth, params.max_depth, params.target_daily_vol)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Thresholds and volatility target must be finite")
        if not 0 < params.min_depth < params.max_depth:
            raise ValueError("Require 0 < min_depth < max_depth")
        if params.target_daily_vol <= 0:
            raise ValueError("target_daily_vol must be positive")
        return int(max(params.peak_window + 2, params.vol_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["damage"] = out["drawdown"].diff()
        out["previous_damage"] = out["damage"].shift(1)
        out["daily_vol"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out["scaled_depth"] = out["drawdown"] / out["daily_vol"].where(out["daily_vol"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["eligible"] = (
            ready
            & out["scaled_depth"].between(params.min_depth, params.max_depth)
            & (out["damage"] > 0)
            & (out["damage"] < out["previous_damage"])
        )
        return out.fillna(0.0)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Stay long precisely while the entry condition holds; its reversal exits.
        df["signal"] = indicators["eligible"].astype(int)
        vol = indicators["daily_vol"].where(indicators["daily_vol"] > 0)
        df["size"] = (params.target_daily_vol / vol).clip(lower=0.01, upper=0.95).fillna(0.01)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 99.0, 98.0, 97.5, 98.0, 98.0, 98.0],
        index=pd.date_range("2020-01-01", periods=10),
    )
    data = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1000.0})
    params = DrawdownParams(peak_window=4, vol_window=3, min_depth=0.1, max_depth=10.0)
    features = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert GeneratedStrategy.warmup_bars(DrawdownParams()) == 10
    assert features["eligible"].iloc[6] and not features["eligible"].iloc[7]
    assert signals["signal"].iloc[7] == 1 and signals["signal"].iloc[8] == 0
    assert signals["signal"].equals(features["eligible"].astype(int).shift(1).fillna(0).astype(int))
    assert signals["signal"].isin([0, 1]).all()
    assert np.isfinite(signals["size"]).all() and (signals["size"] > 0).all()
