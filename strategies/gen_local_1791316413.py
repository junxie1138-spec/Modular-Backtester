from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_bars: int = 20
    volume_bars: int = 20
    pressure_bars: int = 4
    overflow_threshold: float = 0.06


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791316413"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, max(params.channel_bars, params.volume_bars) + params.pressure_bars)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.channel_bars, params.volume_bars, params.pressure_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.overflow_threshold) or params.overflow_threshold <= 0:
            raise ValueError("overflow_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["floor"] = data["low"].rolling(params.channel_bars).min().shift(1)
        out["ceiling"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["width"] = (out["ceiling"] - out["floor"]).where(lambda x: x > 0)
        out["volume_capacity"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["volume_capacity"].where(lambda x: x > 0)
        out["excess_volume"] = (out["volume_ratio"] - 1.0).clip(lower=0.0)
        out["position"] = (data["close"] - out["floor"]) / out["width"]
        out["migration"] = data["close"].diff() / out["width"]
        out["overflow"] = out["migration"] * out["excess_volume"]
        out["pressure"] = out["overflow"].rolling(params.pressure_bars).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        valid = indicators[["ma200", "position", "pressure"]].notna().all(axis=1)
        bullish = valid & (indicators["pressure"] >= params.overflow_threshold) & (indicators["position"] > 0.55)
        bearish = valid & (indicators["pressure"] <= -params.overflow_threshold) & (indicators["position"] < 0.45)
        direction = np.select([bullish, bearish], [1, -1], default=0)
        regime = np.sign(data["close"] - indicators["ma200"]).fillna(0).to_numpy()
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction):
            if position != 0:
                # Only an opposite directional entry condition closes a holding.
                if candidate == -position:
                    position = 0
            elif candidate != 0 and candidate == regime[i]:
                position = int(candidate)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [101, 101, 101, 99, 99, 99]}, index=index)
    derived = pd.DataFrame({
        "ma200": [100.0] * 6,
        "position": [0.8, 0.5, 0.2, 0.2, 0.5, 0.8],
        "pressure": [0.1, 0.0, -0.1, -0.1, 0.0, 0.1],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result.data["size"] > 0).all()
