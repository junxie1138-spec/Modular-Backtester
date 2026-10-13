from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    fractional_order: float = 0.55
    breakout_bars: int = 3
    drought_ratio: float = 0.90
    revival_ratio: float = 1.15
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local-w2_1790989897"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        lookback = min(6, max(2, int(params.breakout_bars)))
        return max(6, 4 + lookback)

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume > 0.0)
        order = min(0.95, max(0.05, float(params.fractional_order)))
        lookback = min(6, max(2, int(params.breakout_bars)))
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, 4):
            weight *= (lag - 1.0 - order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["upper"] = fractional_close.shift(1).rolling(lookback, min_periods=lookback).max()
        out["lower"] = fractional_close.shift(1).rolling(lookback, min_periods=lookback).min()
        out["recent_volume"] = volume.shift(1).rolling(2, min_periods=2).mean()
        out["earlier_volume"] = volume.shift(3).rolling(2, min_periods=2).mean()
        out["recent_volume_peak"] = volume.shift(1).rolling(2, min_periods=2).max()
        out["volume"] = volume
        out["drought"] = out["recent_volume"] < float(params.drought_ratio) * out["earlier_volume"]
        out["revival"] = out["volume"] > float(params.revival_ratio) * out["recent_volume_peak"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        ready = indicators[["fractional_close", "upper", "lower", "recent_volume", "earlier_volume", "recent_volume_peak", "volume"]].notna().all(axis=1)
        participation = ready & indicators["drought"] & indicators["revival"]
        entries = np.zeros(len(data), dtype=int)
        entries[(participation & (indicators["fractional_close"] > indicators["upper"])).to_numpy()] = 1
        entries[(participation & (indicators["fractional_close"] < indicators["lower"])).to_numpy()] = -1
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        target = max(0.0001, float(params.profit_target))
        holding_limit = min(2, max(1, int(params.hold_bars)))
        position = 0
        entry_price = 0.0
        age = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position != 0:
                if entry_price == 0.0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        age = 0
                        continue
                    entry_price = opens[i]
                age += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_close else 0.0
                if not valid_close or gain >= target or age >= holding_limit:
                    position = 0
                    entry_price = 0.0
                    age = 0
                    continue
                decisions[i] = position
            elif i >= warmup - 1 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_price = 0.0
                age = 0
                decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
