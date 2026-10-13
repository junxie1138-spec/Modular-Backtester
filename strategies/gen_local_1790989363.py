from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AssimilationParams:
    innovation_threshold: float = 0.015
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AssimilationParams]):
    strategy_id = "gen_local_1790989363"

    @classmethod
    def params_type(cls) -> type[AssimilationParams]:
        return AssimilationParams

    def warmup_bars(self, params: AssimilationParams) -> int:
        return 64

    def indicators(self, data: pd.DataFrame, params: AssimilationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean().where(lambda x: x > 0.0)
        out["close_level"] = close.ewm(span=16, adjust=False, min_periods=16).mean()
        out["midpoint_level"] = out["midpoint"].ewm(span=16, adjust=False, min_periods=16).mean()
        out["close_slope"] = out["close_level"].diff().ewm(span=4, adjust=False, min_periods=4).mean()
        out["midpoint_slope"] = out["midpoint_level"].diff().ewm(span=4, adjust=False, min_periods=4).mean()
        out["close_strength"] = out["close_slope"] / out["atr"]
        out["midpoint_strength"] = out["midpoint_slope"] / out["atr"]
        out["innovation"] = out["close_slope"].diff() / out["atr"]
        out["slope_gap"] = (out["midpoint_slope"] - out["close_slope"]) / out["atr"]
        out["previous_gap"] = out["slope_gap"].shift(1)
        out["direction"] = np.sign(out["midpoint_slope"])
        out = out.replace([np.inf, -np.inf], np.nan)
        direction = out["direction"]
        threshold = max(float(params.innovation_threshold), 0.0)
        out["eligible"] = (
            (direction * out["close_strength"] > 0.03)
            & (direction * out["midpoint_strength"] > 0.03)
            & (direction * out["innovation"] > threshold)
            & (direction * out["previous_gap"] > threshold)
            & (out["slope_gap"].abs() < out["previous_gap"].abs())
        ).fillna(False)
        out["entry"] = out["eligible"] & ~out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AssimilationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry_price = np.nan
        stop_distance = 0.0
        entry_bar = -1
        stop_multiplier = max(float(params.stop_atr), 0.01)
        for i in range(count):
            # A newly requested position fills at this bar's open.
            if side != 0 and not np.isfinite(entry_price):
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    side = 0
                    signals[i] = 0
                    continue
                entry_price = opens[i]
                entry_bar = i
            if side != 0:
                # Entry ATR is frozen; the stop never trails.
                stop_hit = np.isfinite(closes[i]) and side * (closes[i] - entry_price) <= -stop_distance
                expired = i - entry_bar + 1 >= 8
                if stop_hit or expired or not np.isfinite(closes[i]):
                    side = 0
                    entry_price = np.nan
            elif i >= self.warmup_bars(params) - 1:
                if entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                    side = int(directions[i])
                    stop_distance = stop_multiplier * atr[i]
                    entry_price = np.nan
            signals[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
