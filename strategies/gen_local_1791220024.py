from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_threshold: float = 0.15
    stop_atr: float = 1.8

    def __post_init__(self):
        if not np.isfinite(self.area_threshold) or self.area_threshold <= 0:
            raise ValueError("area_threshold must be finite and positive")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791220024"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return 14

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        r = out["log_return"]
        out["path_variation"] = r.abs().rolling(6, min_periods=6).sum()
        # Antisymmetric level-two signature of six (time, log-price) increments.
        out["area_numerator"] = sum((2 * j - 5) * r.shift(5 - j) for j in range(6))
        out["area"] = out["area_numerator"] / (12.0 * out["path_variation"].replace(0, np.nan))
        out["lag_area"] = out["area"].shift(6)
        out["area_change"] = out["area"] - out["lag_area"]
        out["prior_change"] = out["area_change"].shift(1)
        out["block_return"] = r.rolling(6, min_periods=6).sum()
        out["efficiency"] = out["block_return"].abs() / out["path_variation"].replace(0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        change = indicators["area_change"]
        prior = indicators["prior_change"]
        area = indicators["area"]
        drift = indicators["block_return"]
        efficient = indicators["efficiency"] >= 0.25
        long_entry = (change > params.area_threshold) & (prior <= params.area_threshold) & (area > 0) & (drift > 0) & efficient
        short_entry = (change < -params.area_threshold) & (prior >= -params.area_threshold) & (area < 0) & (drift < 0) & efficient
        entry = np.select([long_entry, short_entry], [1, -1], default=0)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, stop = np.nan, np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position, age = 0, 0
                high_water, stop = np.nan, np.nan
                continue
            if position:
                age += 1
                high_water = max(high_water, close[i])
                # Required highest-close trail applies to both directions.
                stop = max(stop, high_water - params.stop_atr * atr[i])
                if close[i] <= stop or age >= 4:
                    position, age = 0, 0
                    high_water, stop = np.nan, np.nan
            elif entry[i]:
                position = int(entry[i])
                age = 0
                high_water = close[i]
                stop = high_water - params.stop_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = SignatureParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 101., 105., 102., 101., 101.]}, index=index)
    ind = pd.DataFrame({
        "area_change": [0.3, 0., 0., 0., 0., 0.],
        "prior_change": 0.,
        "area": 0.2,
        "block_return": 0.01,
        "efficiency": 0.5,
        "atr": [1., 1., 1., 4., 1., 1.]
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    data["close"] = 100.
    ind["atr"] = 1.
    ind[["area_change", "prior_change", "area", "block_return"]] *= -1
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0]
    ind["atr"] = np.nan
    assert GeneratedStrategy.generate_signals(None, data, ind, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
