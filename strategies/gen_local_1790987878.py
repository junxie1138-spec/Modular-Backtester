from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    capacity: float = 4.0
    profit_target: float = 0.035


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790987878"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 46

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(40, min_periods=40).mean()
        out["std"] = close.rolling(40, min_periods=40).std(ddof=0).replace(0.0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["z_change"] = out["z"].diff()
        out["down_excess"] = (-out["z"] - 1.0).clip(lower=0.0)
        out["up_excess"] = (out["z"] - 1.0).clip(lower=0.0)
        out["down_load"] = out["down_excess"].rolling(6, min_periods=6).sum()
        out["up_load"] = out["up_excess"].rolling(6, min_periods=6).sum()
        out["previous_down_load"] = out["down_load"].shift(1)
        out["previous_up_load"] = out["up_load"].shift(1)
        out["peak"] = close.rolling(40, min_periods=40).max()
        out["trough"] = close.rolling(40, min_periods=40).min().replace(0.0, np.nan)
        out["drawdown"] = close / out["peak"].replace(0.0, np.nan) - 1.0
        out["rally"] = close / out["trough"] - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if not np.isfinite(params.capacity) or params.capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be positive and finite")

        ready = indicators.notna().all(axis=1)
        long_entry = (
            ready
            & (indicators["previous_down_load"] > params.capacity)
            & (indicators["down_load"] <= params.capacity)
            & (indicators["z"] < -0.25)
            & (indicators["z_change"] > 0.0)
            & (indicators["drawdown"] <= -0.03)
        ).fillna(False).to_numpy(dtype=bool)
        short_entry = (
            ready
            & (indicators["previous_up_load"] > params.capacity)
            & (indicators["up_load"] <= params.capacity)
            & (indicators["z"] > 0.25)
            & (indicators["z_change"] < 0.0)
            & (indicators["rally"] >= 0.03)
        ).fillna(False).to_numpy(dtype=bool)

        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        entry_price = np.nan
        entry_bar = -1

        for i in range(len(data)):
            # A decision from yesterday fills at today's open.
            if pending_entry:
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False

            if position != 0:
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                profit_hit = (
                    valid_entry
                    and valid_close
                    and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                )
                time_hit = i - entry_bar + 1 >= 18
                if profit_hit or time_hit or not valid_entry:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                if long_entry[i]:
                    position = 1
                    pending_entry = True
                elif short_entry[i]:
                    position = -1
                    pending_entry = True

            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
