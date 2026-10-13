from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRecordParams:
    peak_lookback: int = 126
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownRecordParams]):
    strategy_id = "gen_local_1791299914"

    @classmethod
    def params_type(cls) -> type[DrawdownRecordParams]:
        return DrawdownRecordParams

    @classmethod
    def warmup_bars(cls, params: DrawdownRecordParams) -> int:
        return params.peak_lookback + 10

    def indicators(self, data: pd.DataFrame, params: DrawdownRecordParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.peak_lookback, min_periods=params.peak_lookback).max()
        depth = (1.0 - data["close"] / peak).clip(lower=0.0)
        prior_high = depth.shift(1).rolling(10, min_periods=10).max()
        prior_low = depth.shift(1).rolling(10, min_periods=10).min()
        return pd.DataFrame({
            "peak": peak.fillna(0.0),
            "depth": depth.fillna(0.0),
            "prior_high": prior_high.fillna(0.0),
            "prior_low": prior_low.fillna(0.0),
            "close_change": data["close"].diff().fillna(0.0),
            "ready": prior_high.notna() & prior_low.notna(),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownRecordParams,
    ) -> SignalFrame:
        depth = indicators["depth"]
        shallow_record = (
            (depth >= 0.02) & (depth < 0.08)
            & (depth > indicators["prior_high"])
            & (indicators["close_change"] < 0.0)
        )
        deep_recovery = (
            (depth >= 0.08)
            & (depth < indicators["prior_low"])
            & (indicators["close_change"] > 0.0)
        )
        entries = (indicators["ready"] & (shallow_record | deep_recovery)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if i == exit_at:
                exit_at = -1
            elif exit_at < 0 and enter:
                exit_at = i + params.hold_bars
            if exit_at > i:
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
