from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapOccupancyParams:
    gap_window: int = 18
    position_window: int = 18
    gap_floor: float = 0.0005
    agreement: float = 0.22
    position_bias: float = 0.08


class GeneratedStrategy(BaseStrategy[GapOccupancyParams]):
    strategy_id = "gen_local_1791249090"

    @classmethod
    def params_type(cls) -> type[GapOccupancyParams]:
        return GapOccupancyParams

    def warmup_bars(self, params: GapOccupancyParams) -> int:
        return int(max(params.gap_window + 1, params.position_window))

    def indicators(self, data: pd.DataFrame, params: GapOccupancyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        out["gap"] = (data["open"] / prior_close.where(prior_close > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_vote"] = np.sign(out["gap"]).where(out["gap"].abs() >= params.gap_floor, 0.0).where(out["gap"].notna())
        out["gap_balance"] = out["gap_vote"].rolling(params.gap_window, min_periods=params.gap_window).mean()
        out["channel_floor"] = data["low"].rolling(params.position_window, min_periods=params.position_window).min()
        out["channel_ceiling"] = data["high"].rolling(params.position_window, min_periods=params.position_window).max()
        out["mean_close"] = data["close"].rolling(params.position_window, min_periods=params.position_window).mean()
        out["channel_span"] = out["channel_ceiling"] - out["channel_floor"]
        out["occupancy"] = ((out["mean_close"] - out["channel_floor"]) / out["channel_span"].where(out["channel_span"] > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapOccupancyParams) -> SignalFrame:
        balance = indicators["gap_balance"]
        occupancy = indicators["occupancy"]
        ready = np.isfinite(balance) & np.isfinite(occupancy)
        long_entry = ready & (balance >= params.agreement) & (occupancy >= 0.5 + params.position_bias)
        short_entry = ready & (balance <= -params.agreement) & (occupancy <= 0.5 - params.position_bias)
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapOccupancyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"gap_balance": [0.5, np.nan, -0.5, -0.5, 0.0, 0.5, 0.5], "occupancy": [0.8, np.nan, 0.2, 0.2, 0.5, 0.8, 0.8]}, index=index)
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) == 19
    flat_evidence = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat_evidence, None, params).data["signal"].eq(0).all()
