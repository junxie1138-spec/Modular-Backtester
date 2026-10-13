from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AllocationParams:
    volume_window: int = 20
    volume_multiple: float = 1.15
    minimum_return: float = 0.003
    minimum_close_location: float = 0.70
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AllocationParams]):
    strategy_id = "gen_local-w2_1790982226"

    @classmethod
    def params_type(cls) -> type[AllocationParams]:
        return AllocationParams

    @staticmethod
    def warmup_bars(params: AllocationParams) -> int:
        return int(max(200, params.volume_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AllocationParams) -> pd.DataFrame:
        if params.volume_window < 1 or params.max_hold_bars < 1:
            raise ValueError("Lookbacks and holding periods must be positive")
        if params.volume_multiple <= 0 or params.profit_target <= 0:
            raise ValueError("Volume multiple and profit target must be positive")
        if not 0 <= params.minimum_close_location <= 1:
            raise ValueError("Close location must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = data["close"].shift(1)
        out["daily_return"] = data["close"].div(out["previous_close"].where(out["previous_close"] > 0)).sub(1)
        out["prior_volume_mean"] = data["volume"].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out["volume_ratio"] = data["volume"].div(out["prior_volume_mean"].where(out["prior_volume_mean"] > 0))
        out["daily_range"] = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]).div(out["daily_range"].where(out["daily_range"] > 0))
        out["days_to_month_end"] = pd.Series(data.index.days_in_month - data.index.day, index=data.index)
        out["entry"] = (
            (out["days_to_month_end"] <= 4)
            & (data["close"] > out["ma200"])
            & (out["daily_return"] >= params.minimum_return)
            & (out["volume_ratio"] >= params.volume_multiple)
            & (out["close_location"] >= params.minimum_close_location)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AllocationParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    continue
            if active:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_expired:
                    active = False
                    continue
                decisions[i] = 1
            elif entries[i] and i + 1 < len(data):
                decisions[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
