from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    window: int = 8
    area_threshold: float = 0.08
    stretch: float = 0.15

    def __post_init__(self):
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if not np.isfinite(self.area_threshold) or not 0 < self.area_threshold < 0.5:
            raise ValueError("area_threshold must be between 0 and 0.5")
        if not np.isfinite(self.stretch) or not 0 < self.stretch < 1:
            raise ValueError("stretch must be between 0 and 1")


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790854987"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(params.window + 1)

    def indicators(self, data, params):
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        activity = data["volume"].where(data["volume"] >= 0)
        price_path = pd.Series(0.0, index=data.index)
        volume_path = price_path.copy()
        variation = price_path.copy()
        area = price_path.copy()
        # Level-two antisymmetric signature of (cumulative volume, log price).
        for lag in range(params.window - 1, -1, -1):
            dx = returns.shift(lag)
            dv = activity.shift(lag)
            area = area + 0.5 * (volume_path * dx - price_path * dv)
            price_path = price_path + dx
            volume_path = volume_path + dv
            variation = variation + dx.abs()
        denominator = (volume_path * variation).where(
            (volume_path > 0) & (variation > 0)
        )
        score = area / denominator
        stretch = price_path / variation.where(variation > 0)
        recoil = returns.rolling(2, min_periods=2).sum()
        result = pd.DataFrame({
            "returns": returns,
            "activity": activity,
            "price_path": price_path,
            "volume_path": volume_path,
            "variation": variation,
            "area": area,
            "score": score,
            "stretch": stretch,
            "recoil": recoil,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        result["ready"] = result[["score", "stretch", "recoil"]].notna().all(axis=1)
        return result.fillna(0.0)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators["ready"].astype(bool)
        long_entry = (
            ready
            & (indicators["score"] >= params.area_threshold)
            & (indicators["stretch"] <= -params.stretch)
            & (indicators["recoil"] > 0)
        )
        short_entry = (
            ready
            & (indicators["score"] <= -params.area_threshold)
            & (indicators["stretch"] >= params.stretch)
            & (indicators["recoil"] < 0)
        )
        intent = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(intent):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    direction = np.array([1, 0, 0, -1, -1, 0, 1])
    indicators = pd.DataFrame({
        "ready": True,
        "score": direction * 0.2,
        "stretch": direction * -0.4,
        "recoil": direction * 0.01,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1]
    mirrored = indicators.copy()
    mirrored[["score", "stretch", "recoil"]] *= -1
    reverse = strategy.generate_signals(data, mirrored, ctx, params).data
    assert reverse["signal"].equals(-signals["signal"])
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
