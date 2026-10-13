from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    extension_threshold: float = 0.12
    minimum_overlap: float = 0.30


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791292380"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not 0.0 < params.extension_threshold < 1.0:
            raise ValueError("extension_threshold must be between zero and one")
        if not 0.0 < params.minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must be positive and at most one")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["previous_range"] = out["previous_high"] - out["previous_low"]
        out["upper_extension"] = (data["high"] - out["previous_high"]).clip(lower=0.0)
        out["lower_extension"] = (out["previous_low"] - data["low"]).clip(lower=0.0)
        out["overlap_high"] = np.minimum(data["high"], out["previous_high"])
        out["overlap_low"] = np.maximum(data["low"], out["previous_low"])
        out["overlap_width"] = (out["overlap_high"] - out["overlap_low"]).clip(lower=0.0)
        out["smaller_range"] = np.minimum(out["range"], out["previous_range"])
        out["combined_range"] = out["range"] + out["previous_range"]
        out["overlap_fraction"] = out["overlap_width"] / out["smaller_range"].where(out["smaller_range"] > 0)
        out["overlap_position"] = (data["close"] - out["overlap_low"]) / out["overlap_width"].where(out["overlap_width"] > 0)
        out["rejection_balance"] = (out["lower_extension"] - out["upper_extension"]) / out["combined_range"].where(out["combined_range"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        eligible = (indicators["overlap_fraction"] >= params.minimum_overlap) & indicators["overlap_position"].between(0.0, 1.0)
        buy = eligible & (indicators["rejection_balance"] >= params.extension_threshold) & (indicators["overlap_position"] > 0.5)
        sell = eligible & (indicators["rejection_balance"] <= -params.extension_threshold) & (indicators["overlap_position"] < 0.5)
        evidence = np.select([buy, sell], [1, -1], default=0)
        regime = np.select([data["close"] > indicators["ma200"], data["close"] < indicators["ma200"]], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(evidence):
            if position:
                # Opposing setup exits even if the MA regime blocks a new entry.
                if direction == -position:
                    position = 0
            elif direction != 0 and direction == regime[i]:
                position = int(direction)
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=205)
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    derived = strategy.indicators(flat, params)
    assert derived["ma200"].iloc[:199].isna().all()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()
    data = flat.iloc[:7]
    probe = pd.DataFrame({"overlap_fraction": 1.0, "overlap_position": [0.75, 0.5, 0.25, 0.25, 0.5, 0.75, 0.5], "rejection_balance": [0.3, 0.0, -0.3, -0.3, 0.0, 0.3, 0.0], "ma200": [99.0, 99.0, 99.0, 101.0, 101.0, 101.0, 99.0]}, index=data.index)
    assert strategy.generate_signals(data, probe, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]


if __name__ == "__main__":
    _self_check()
