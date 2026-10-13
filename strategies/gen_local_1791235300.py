from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalWaveParams:
    fractional_order: float = 0.4
    stop_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[FractionalWaveParams]):
    strategy_id = "gen_local_1791235300"

    @classmethod
    def params_type(cls) -> type[FractionalWaveParams]:
        return FractionalWaveParams

    def warmup_bars(self, params: FractionalWaveParams) -> int:
        return 54

    def indicators(self, data: pd.DataFrame, params: FractionalWaveParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = [1.0]
        for lag in range(1, 32):
            weights.append(-weights[-1] * (params.fractional_order - lag + 1) / lag)
        # ponytail: fixed 32-bar memory; extend the kernel if truncation dominates results.
        fractional_close = close * weights[0]
        for lag in range(1, 32):
            fractional_close = fractional_close + close.shift(lag) * weights[lag]
        out["fractional_close"] = fractional_close
        out["ceiling"] = fractional_close.shift(1).rolling(20, min_periods=20).max()
        out["impulse"] = fractional_close.diff()
        valid_pair = out["impulse"].notna() & out["impulse"].shift(1).notna()
        out["reversal"] = (out["impulse"] * out["impulse"].shift(1) < 0.0).astype(float).where(valid_pair)
        out["prior_reversals"] = out["reversal"].shift(1).rolling(6, min_periods=6).sum()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["fractional_close"] > out["ceiling"])
            & (out["fractional_close"].shift(1) <= out["ceiling"].shift(1))
            & (out["prior_reversals"] >= 3.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalWaveParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if active:
                if i == entry_bar:
                    # The preceding close requested entry at this session's open.
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                held_sessions = i - entry_bar + 1
                if (
                    not np.isfinite(fixed_stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < fixed_stop
                    or held_sessions >= 4
                ):
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and np.isfinite(closes[i]):
                active = True
                entry_bar = i + 1
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
