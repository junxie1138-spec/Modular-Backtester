from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    fast_window: int = 4
    rear_window: int = 12
    volatility_window: int = 40
    acceleration_threshold: float = 0.15
    profit_target: float = 0.05
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = "gen_local_1790992118"

    @classmethod
    def params_type(cls) -> type[PropagationParams]:
        return PropagationParams

    def warmup_bars(self, params: PropagationParams) -> int:
        fast = max(1, int(params.fast_window))
        rear = max(1, int(params.rear_window))
        vol = max(2, int(params.volatility_window))
        return int(max(3 * fast + 1, rear + 2 * fast + 1, vol + 1))

    def indicators(self, data: pd.DataFrame, params: PropagationParams) -> pd.DataFrame:
        fast = max(1, int(params.fast_window))
        rear = max(1, int(params.rear_window))
        vol_window = max(2, int(params.volatility_window))
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None)
        front_roc = close.pct_change(fast, fill_method=None) / fast
        rear_roc = close.shift(fast).pct_change(rear, fill_method=None) / rear
        front_acceleration = front_roc.diff(fast)
        rear_acceleration = rear_roc.diff(fast)
        volatility = returns.rolling(vol_window, min_periods=vol_window).std(ddof=0)
        scale = volatility.where(volatility > 0)
        out = pd.DataFrame(index=data.index)
        out["front_roc"] = front_roc
        out["rear_roc"] = rear_roc
        out["front_acceleration"] = front_acceleration
        out["prior_front_acceleration"] = front_acceleration.shift(fast)
        out["rear_acceleration"] = rear_acceleration
        out["volatility"] = volatility
        out["initiating_strength"] = front_acceleration.shift(fast) / scale
        out["transmitted_strength"] = rear_acceleration / scale
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out.notna().all(axis=1) & (volatility > 0)
        out["strength"] = out[["initiating_strength", "transmitted_strength"]].min(axis=1).clip(lower=0)
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PropagationParams,
    ) -> SignalFrame:
        ind = indicators
        eligible = (
            ind["ready"].astype(bool)
            & (ind["initiating_strength"] >= params.acceleration_threshold)
            & (ind["transmitted_strength"] >= params.acceleration_threshold)
            & (ind["front_roc"] > ind["rear_roc"])
            & (ind["rear_roc"] > 0)
            & (ind["front_acceleration"] < ind["prior_front_acceleration"])
        )
        entry = eligible & ~eligible.shift(1, fill_value=False)
        entry &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        entry_flags = entry.to_numpy(dtype=bool)
        strengths = ind["strength"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        max_hold = max(1, int(params.max_hold_bars))
        target = max(0.0, float(params.profit_target))

        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + target)
                time_expired = i - entry_bar + 1 >= max_hold
                if not valid_entry or target_hit or time_expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
                    sizes[i] = held_size
            elif entry_flags[i]:
                entry_bar = i + 1
                held_size = float(0.25 + 0.70 * np.clip(strengths[i] / 0.60, 0.0, 1.0))
                signals[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
