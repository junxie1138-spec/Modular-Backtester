from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    shock_multiple: float = 1.4
    max_overflow: float = 0.15
    minimum_drawdown: float = 0.015
    profit_target: float = 0.025
    max_hold_bars: int = 8
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791253918"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["prior_range"] = out["range"].shift(1)
        out["baseline_range"] = out["range"].rolling(4, min_periods=4).mean().shift(2)
        out["shock_ratio"] = out["prior_range"] / out["baseline_range"]
        out["prior_peak"] = data["high"].rolling(6, min_periods=6).max().shift(1)
        out["drawdown"] = 1.0 - data["close"] / out["prior_peak"].where(out["prior_peak"] > 0)
        out["overflow"] = (data["low"].shift(1) - data["low"]) / out["prior_range"]
        out["prior_midpoint"] = (data["high"].shift(1) + data["low"].shift(1)) / 2.0
        out["prior_down_bar"] = data["close"].shift(1) < data["open"].shift(1)
        out["entry"] = (
            (out["shock_ratio"] >= params.shock_multiple)
            & (out["drawdown"] >= params.minimum_drawdown)
            & (out["overflow"] > 0.0)
            & (out["overflow"] <= params.max_overflow)
            & (out["range"] <= out["prior_range"])
            & (data["close"] > out["prior_midpoint"])
            & (data["close"] > data["open"])
            & out["prior_down_bar"]
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not (0.0 < params.size <= 1.0):
            raise ValueError("size must be in (0, 1]")
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("profit_target and max_hold_bars must be positive")

        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        entry_price = np.nan
        held_bars = 0

        for i in range(len(data)):
            previous_decision = decisions[i - 1] if i else 0
            if previous_decision == 0:
                entry_price = np.nan
                held_bars = 0
            elif not np.isfinite(entry_price):
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_price = opens[i]

            if np.isfinite(entry_price):
                held_bars += 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_hit = held_bars >= params.max_hold_bars
                decisions[i] = 0 if target_hit or time_hit else 1
            elif previous_decision == 0 and entries[i]:
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
