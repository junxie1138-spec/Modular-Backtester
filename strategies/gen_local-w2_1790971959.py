from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.45
    frac_window: int = 24
    trend_window: int = 4
    capacity_window: int = 20
    strength_floor: float = 0.03
    pressure_floor: float = 0.15
    size: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must lie between zero and one")
        if min(self.frac_window, self.trend_window, self.capacity_window) < 2:
            raise ValueError("windows must be at least two")
        if not all(np.isfinite(x) for x in (self.strength_floor, self.pressure_floor, self.size)):
            raise ValueError("thresholds and size must be finite")
        if min(self.strength_floor, self.pressure_floor) < 0.0 or self.size <= 0.0:
            raise ValueError("thresholds must be nonnegative and size positive")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790971959"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.frac_window + params.trend_window + 1,
                   params.capacity_window + params.trend_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].astype(float).where(data["volume"] >= 0.0)
        volume = volume.replace([np.inf, -np.inf], np.nan)
        # Fixed finite-memory fractional difference; weights multiply past closes.
        frac_close = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            frac_close = frac_close + weight * close.shift(lag)
        out["frac_close"] = frac_close
        span = (data["high"] - data["low"]).where(data["high"] > data["low"])
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["range_capacity"] = out["true_range"].rolling(params.capacity_window).mean()
        out["volume_capacity"] = volume.shift(1).rolling(params.capacity_window).median()
        out["close_location"] = ((2.0 * close - data["high"] - data["low"]) / span).clip(-1.0, 1.0)
        out["signed_volume"] = out["close_location"] * volume
        out["normalized_pressure"] = out["signed_volume"] / out["volume_capacity"].where(out["volume_capacity"] > 0.0)
        out["flow_pressure"] = out["normalized_pressure"].rolling(params.trend_window).mean()
        out["frac_strength"] = (
            out["frac_close"].diff(params.trend_window) / params.trend_window
            / out["range_capacity"].where(out["range_capacity"] > 0.0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Stay long only while BOTH primitives agree; either reversal exits.
        entry = (
            (indicators["frac_strength"] > params.strength_floor)
            & (indicators["flow_pressure"] > params.pressure_floor)
        ).fillna(False)
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80)
    close = pd.Series(100.0 + np.arange(80) * 0.2, index=index)
    data = pd.DataFrame({"open": close - 0.1, "high": close + 0.3,
                         "low": close - 0.7, "close": close, "volume": 1000.0})
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(derived.iloc[:50], strategy.indicators(data.iloc[:50], params))
    assert derived["frac_strength"].iloc[:params.frac_window + params.trend_window - 1].isna().all()
    assert np.isfinite(derived.iloc[strategy.warmup_bars(params):].to_numpy()).all()
    forced = pd.DataFrame({"frac_strength": [np.nan, 1.0, 1.0, 0.0, 1.0],
                           "flow_pressure": [np.nan, 1.0, 0.0, 1.0, 1.0]}, index=index[:5])
    result = strategy.generate_signals(data.iloc[:5], forced, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0]
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
