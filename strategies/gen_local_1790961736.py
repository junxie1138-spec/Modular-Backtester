from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    window: int = 8
    pressure_threshold: float = 0.45
    distance_threshold: float = 0.65

    def __post_init__(self):
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not 0.0 < self.pressure_threshold < 1.0:
            raise ValueError("pressure_threshold must be between 0 and 1")
        if not np.isfinite(self.distance_threshold) or self.distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be finite and positive")


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = "gen_local_1790961736"

    @classmethod
    def params_type(cls):
        return ImpactParams

    def warmup_bars(self, params):
        return int(max(200, params.window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["move"] = data["close"].diff()
        out["impact"] = data["volume"].clip(lower=0) * out["move"].abs()
        out["signed_impact"] = data["volume"].clip(lower=0) * out["move"]
        out["impact_price"] = out["impact"] * data["close"]
        out["impact_sum"] = out["impact"].rolling(params.window, min_periods=params.window).sum()
        out["signed_sum"] = out["signed_impact"].rolling(params.window, min_periods=params.window).sum()
        out["price_sum"] = out["impact_price"].rolling(params.window, min_periods=params.window).sum()
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_mean"] = out["range"].rolling(params.window, min_periods=params.window).mean()
        denominator = out["impact_sum"].where(out["impact_sum"] > 0)
        out["reference"] = out["price_sum"] / denominator
        out["pressure"] = out["signed_sum"] / denominator
        out["displacement"] = (data["close"] - out["reference"]) / out["range_mean"].where(out["range_mean"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        ready = np.isfinite(indicators[["ma200", "pressure", "displacement"]]).all(axis=1)
        entry = ready & (data["close"] > indicators["ma200"]) & (indicators["pressure"] <= -params.pressure_threshold) & (indicators["displacement"] <= -params.distance_threshold)
        reversal = ready & (indicators["pressure"] >= params.pressure_threshold) & (indicators["displacement"] >= params.distance_threshold)
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(np.where(entry, 1.0, np.where(reversal, 0.0, np.nan)), index=data.index).ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=204)
    data = pd.DataFrame({"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1000.0}, index=index)
    params = ImpactParams()
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    assert indicators["ma200"].iloc[:199].isna().all()
    indicators.loc[index[200:], "pressure"] = [-0.6, 0.0, 0.6, 0.0]
    indicators.loc[index[200:], "displacement"] = [-1.0, 0.0, 1.0, 0.0]
    indicators.loc[index[200:], "ma200"] = 100.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[-4:].tolist() == [0, 1, 1, 0]
    assert result["size"].gt(0).all()
    indicators.loc[index[200], "ma200"] = 102.0
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(ImpactParams(window=250)) == 251


if __name__ == "__main__":
    _self_check()
