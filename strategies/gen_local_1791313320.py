from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoveryParams:
    lookback: int = 5
    range_expansion: float = 1.5
    contraction: float = 0.65
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeRecoveryParams]):
    strategy_id = "gen_local_1791313320"

    @classmethod
    def params_type(cls):
        return RangeRecoveryParams

    def warmup_bars(self, params: RangeRecoveryParams) -> int:
        if not 2 <= params.lookback <= 8:
            raise ValueError("lookback must be between 2 and 8")
        if params.range_expansion <= 1 or not 0 < params.contraction < 1:
            raise ValueError("Invalid range thresholds")
        if not 0 < params.profit_target < 1 or params.hold_bars not in (1, 2):
            raise ValueError("Invalid exit parameters")
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RangeRecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["shock_range"] = out["range"].shift(1)
        out["baseline_range"] = out["range"].rolling(params.lookback, min_periods=params.lookback).mean().shift(2)
        out["prior_floor"] = data["close"].rolling(params.lookback, min_periods=params.lookback).min().shift(2)
        out["prior_ceiling"] = data["close"].rolling(params.lookback, min_periods=params.lookback).max().shift(2)
        out["shock_close"] = data["close"].shift(1)
        out["anchor_close"] = data["close"].shift(2)
        out["shock_midpoint"] = ((data["high"] + data["low"]) / 2.0).shift(1)
        out["expanded"] = out["shock_range"] >= params.range_expansion * out["baseline_range"]
        out["contracted"] = out["range"] <= params.contraction * out["shock_range"]
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1) & (data["close"] > 0)
        ready = out["expanded"] & out["contracted"] & valid
        buy = ready & (out["shock_close"] < out["prior_floor"]) & (data["close"] > out["shock_midpoint"]) & (data["close"] > out["shock_close"]) & (data["close"] < out["anchor_close"])
        sell = ready & (out["shock_close"] > out["prior_ceiling"]) & (data["close"] < out["shock_midpoint"]) & (data["close"] < out["shock_close"]) & (data["close"] > out["anchor_close"])
        out["entry"] = np.select([buy.fillna(False), sell.fillna(False)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeRecoveryParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry_price = None
        for i, candidate in enumerate(entries):
            if position:
                # The previous decision fills at this bar's open.
                if entry_price is None:
                    entry_price = opens[i]
                age += 1
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position, age, entry_price = 0, 0, None
                else:
                    # Close-confirmed targets exit at the following open.
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or age >= params.hold_bars:
                        position, age, entry_price = 0, 0, None
            elif candidate:
                position, age, entry_price = int(candidate), 0, None
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = RangeRecoveryParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) <= 10
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived["entry"].eq(0).all()


if __name__ == "__main__":
    _self_check()
