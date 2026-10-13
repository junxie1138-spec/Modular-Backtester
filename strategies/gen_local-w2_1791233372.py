from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    entry_streak: int = 2
    reversal_streak: int = 4
    excursion_pct: float = 0.015
    excursion_lookback: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791233372"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.excursion_lookback, params.entry_streak + 1, params.reversal_streak + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.entry_streak < 1 or params.reversal_streak < 1:
            raise ValueError("Streak lengths must be positive")
        if params.excursion_lookback < 2 or not 0.0 < params.excursion_pct < 1.0:
            raise ValueError("Invalid excursion parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["change"] = close.diff()
        rising = out["change"].gt(0)
        falling = out["change"].lt(0)
        out["up_streak"] = rising.astype(int).groupby((~rising).cumsum()).cumsum()
        out["down_streak"] = falling.astype(int).groupby((~falling).cumsum()).cumsum()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["recent_peak"] = close.rolling(params.excursion_lookback, min_periods=params.excursion_lookback).max()
        out["recent_trough"] = close.rolling(params.excursion_lookback, min_periods=params.excursion_lookback).min()
        out["drawdown"] = (1.0 - close / out["recent_peak"].where(out["recent_peak"].gt(0))).replace([np.inf, -np.inf], np.nan)
        out["rebound"] = (close / out["recent_trough"].where(out["recent_trough"].gt(0)) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["ready"] = close.gt(0) & out[["ma200", "drawdown", "rebound", "change"]].notna().all(axis=1)
        out["long_entry"] = out["ready"] & close.gt(out["ma200"]) & out["drawdown"].ge(params.excursion_pct) & out["up_streak"].ge(params.entry_streak)
        out["short_entry"] = out["ready"] & close.lt(out["ma200"]) & out["rebound"].ge(params.excursion_pct) & out["down_streak"].ge(params.entry_streak)
        out["bearish_reversal"] = out["ready"] & out["down_streak"].ge(params.reversal_streak)
        out["bullish_reversal"] = out["ready"] & out["up_streak"].ge(params.reversal_streak)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        events = indicators[["long_entry", "short_entry", "bearish_reversal", "bullish_reversal"]].fillna(False).to_numpy(dtype=bool)
        for i, (long_entry, short_entry, bearish_reversal, bullish_reversal) in enumerate(events):
            if position == 1:
                if bearish_reversal:
                    position = 0
            elif position == -1:
                if bullish_reversal:
                    position = 0
            elif long_entry:
                position = 1
            elif short_entry:
                position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
