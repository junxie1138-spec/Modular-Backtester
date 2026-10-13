from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 6
    extreme_z: float = 1.3
    recovery_z: float = 0.35
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791228682"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        if not 2 <= params.ma_window <= 8:
            raise ValueError("ma_window must be between 2 and 8")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        values = [params.extreme_z, params.recovery_z, params.profit_target]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Thresholds must be finite and positive")
        return int(max(params.ma_window + 2, 4))

    def indicators(self, data, params):
        self.warmup_bars(params)
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        window = close.rolling(params.ma_window, min_periods=params.ma_window)
        out["ma"] = window.mean().shift(1)
        out["std"] = window.std(ddof=0).shift(1).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        out["previous_z"] = out["z"].shift(1)
        out["previous_close"] = close.shift(1)
        out["prior_low"] = data["low"].rolling(3, min_periods=3).min().shift(1)
        out["prior_high"] = data["high"].rolling(3, min_periods=3).max().shift(1)
        out["z_recovery"] = out["z"] - out["previous_z"]
        long_entry = (
            (out["previous_z"] <= -params.extreme_z)
            & (out["z"] < 0)
            & (out["z_recovery"] >= params.recovery_z)
            & (data["low"] < out["prior_low"])
            & (close > out["previous_close"])
        )
        short_entry = (
            (out["previous_z"] >= params.extreme_z)
            & (out["z"] > 0)
            & (out["z_recovery"] <= -params.recovery_z)
            & (data["high"] > out["prior_high"])
            & (close < out["previous_close"])
        )
        out["entry"] = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position, held, entry_price = 0, 0, 0.0
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this session's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i]):
                    position, held = 0, 0
                elif position * (closes[i] / entry_price - 1.0) >= params.profit_target or held >= params.max_hold_bars:
                    position, held = 0, 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position, held = int(entries[i]), 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 99.0,
        "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    flat = data.assign(close=100.0)
    derived = strategy.indicators(flat, params)
    assert derived.index.equals(data.index) and derived["entry"].eq(0).all()
    assert strategy.warmup_bars(params) == 8


if __name__ == "__main__":
    _self_check()
