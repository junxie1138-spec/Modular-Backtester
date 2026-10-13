from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCascadeParams:
    flow_window: int = 7
    expansion_window: int = 3
    percentile_window: int = 63
    tail_quantile: float = 0.8


class GeneratedStrategy(BaseStrategy[RangeCascadeParams]):
    strategy_id = "gen_local_1790978959"

    @classmethod
    def params_type(cls):
        return RangeCascadeParams

    def warmup_bars(self, params: RangeCascadeParams) -> int:
        return int(params.flow_window + params.expansion_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: RangeCascadeParams) -> pd.DataFrame:
        if min(params.flow_window, params.expansion_window, params.percentile_window) < 2:
            raise ValueError("Window lengths must be at least two.")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must be between 0.5 and 1.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["upper_extension"] = data["high"].diff().clip(lower=0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0)
        out["extension_balance"] = out["upper_extension"] - out["lower_extension"]
        out["range_sum"] = out["range"].rolling(params.flow_window).sum()
        out["flow"] = out["extension_balance"].rolling(params.flow_window).sum() / out["range_sum"].replace(0, np.nan)
        out["recent_range"] = out["range"].rolling(params.expansion_window).mean()
        out["baseline_range"] = out["range"].shift(params.expansion_window).rolling(params.flow_window).mean()
        out["expansion"] = out["recent_range"] / out["baseline_range"].replace(0, np.nan)
        past_flow = out["flow"].shift(1).rolling(params.percentile_window)
        out["flow_upper"] = past_flow.quantile(params.tail_quantile)
        out["flow_lower"] = past_flow.quantile(1.0 - params.tail_quantile)
        out["expansion_threshold"] = out["expansion"].shift(1).rolling(params.percentile_window).quantile(params.tail_quantile)
        out["breakout_high"] = data["high"].shift(1).rolling(params.flow_window).max()
        out["breakout_low"] = data["low"].shift(1).rolling(params.flow_window).min()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeCascadeParams) -> SignalFrame:
        expanding = indicators["expansion"] > indicators["expansion_threshold"]
        long_entry = expanding & (indicators["flow"] > indicators["flow_upper"]) & (indicators["flow"] > 0) & (data["close"] > indicators["breakout_high"])
        short_entry = expanding & (indicators["flow"] < indicators["flow_lower"]) & (indicators["flow"] < 0) & (data["close"] < indicators["breakout_low"])
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position and entry == -position:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [2.0, 1.0, -2.0, -2.0, 1.0, 2.0]}, index=index)
    ind = pd.DataFrame({
        "expansion": 2.0, "expansion_threshold": 1.0,
        "flow": [2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        "flow_upper": 1.0, "flow_lower": -1.0,
        "breakout_high": 1.0, "breakout_low": -1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = RangeCascadeParams()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
