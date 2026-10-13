from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecoveryParams:
    complexity_window: int = 12
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[OrdinalRecoveryParams]):
    strategy_id = "gen_local_1790921279"

    @classmethod
    def params_type(cls):
        return OrdinalRecoveryParams

    def warmup_bars(self, params: OrdinalRecoveryParams) -> int:
        return int(max(21, params.complexity_window + 3))

    def indicators(self, data: pd.DataFrame, params: OrdinalRecoveryParams) -> pd.DataFrame:
        if params.complexity_window < 3 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("complexity_window must be >= 3 and stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["change"] = close.diff()
        out["previous_change"] = out["change"].shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_peak"] = close.rolling(20, min_periods=20).max().shift(1)
        out["previous_drawdown"] = previous_close - out["previous_peak"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = pd.Series(np.select([
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index).where(valid)
        window = params.complexity_window
        out["valid_patterns"] = valid.astype(float).rolling(window, min_periods=window).sum()
        collisions = pd.Series(0.0, index=data.index)
        for motif in range(6):
            column = "count_" + str(motif)
            out[column] = out["pattern"].eq(motif).astype(float).rolling(window, min_periods=window).sum()
            collisions = collisions + out[column] * (out[column] - 1.0)
        out["complexity"] = (1.0 - collisions / float(window * (window - 1))).where(out["valid_patterns"].eq(window))
        out["previous_complexity"] = out["complexity"].shift(1)
        out["complexity_expansion"] = out["complexity"] - out["previous_complexity"]
        out["move_atr"] = out["change"].abs() / out["previous_atr"].where(out["previous_atr"] > 0)
        out["recent_spike"] = out["move_atr"].rolling(3, min_periods=3).max()
        eligible = (
            out["previous_drawdown"].lt(-out["previous_atr"])
            & out["previous_complexity"].lt(0.72)
            & out["complexity_expansion"].gt(0.04)
            & out["recent_spike"].le(2.0)
            & out["atr"].gt(0)
        )
        recovery = out["previous_change"].lt(0) & out["change"].gt(0)
        failed_rebound = out["previous_change"].gt(0) & out["change"].lt(0)
        out["entry_direction"] = np.select([eligible & recovery, eligible & failed_rebound], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalRecoveryParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        fixed_distance = np.nan
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                    fixed_distance = params.stop_atr * atr[i - 1]
                valid_position = np.isfinite(entry_price) and np.isfinite(fixed_distance) and fixed_distance > 0 and np.isfinite(closes[i])
                stopped = valid_position and (
                    (position == 1 and closes[i] < entry_price - fixed_distance)
                    or (position == -1 and closes[i] > entry_price + fixed_distance)
                )
                if not valid_position or stopped or i - entry_bar + 1 >= 2:
                    position = 0
            elif i >= warmup - 1 and entries[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
