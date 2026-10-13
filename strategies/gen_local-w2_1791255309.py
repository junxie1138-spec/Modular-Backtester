from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791255309"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return 24

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["range_fraction"] = ((data["high"] - data["low"]) / close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out["baseline_range"] = out["range_fraction"].rolling(20, min_periods=20).mean().shift(1)
        out["range_ratio"] = out["range_fraction"] / out["baseline_range"].where(out["baseline_range"] > 0)
        out["shock_ratio"] = out["range_ratio"].shift(3)
        out["shock_low"] = data["low"].shift(3)
        out["postshock_low"] = data["low"].rolling(3, min_periods=3).min()
        out["bearish_shock"] = close.shift(3) < data["open"].shift(3)
        r = out["range_fraction"]
        out["range_decay"] = (r < r.shift(1)) & (r.shift(1) < r.shift(2)) & (r.shift(2) < r.shift(3))
        out["close_recovery"] = (close > close.shift(1)) & (close > data["open"])
        out["entry"] = (
            (out["shock_ratio"] >= 1.6)
            & out["bearish_shock"]
            & out["range_decay"]
            & (out["range_ratio"] <= 0.65)
            & (out["postshock_low"] >= out["shock_low"])
            & out["close_recovery"]
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if isinstance(params.max_hold_bars, bool) or not isinstance(params.max_hold_bars, int) or params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be a positive integer")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decision = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                if held == 0:
                    entry_price = opens[i]
                held += 1
                # Evaluate the target at the close; execute the exit next open.
                if closes[i] >= entry_price * (1.0 + params.profit_target) or held >= params.max_hold_bars:
                    active = False
                    held = 0
            elif entries[i]:
                active = True
                held = 0
            decision[i] = int(active)
        df = pd.DataFrame({"signal": decision, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    bars = pd.DataFrame({"open": [100.0, 110.0, 110.0, 110.0, 110.0], "high": 115.0, "low": 99.0, "close": [100.0, 110.0, 113.4, 110.0, 110.0], "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    target_exit = GeneratedStrategy.generate_signals(bars, entries, None, ShockwaveParams(profit_target=0.03, max_hold_bars=8))
    assert target_exit.data["signal"].tolist() == [0, 1, 1, 0, 0]
    bars["close"] = 110.0
    time_exit = GeneratedStrategy.generate_signals(bars, entries, None, ShockwaveParams(profit_target=0.03, max_hold_bars=2))
    assert time_exit.data["signal"].tolist() == [0, 1, 1, 0, 0]
