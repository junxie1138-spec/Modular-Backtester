from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryParams:
    fractional_order: float = 0.5
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[MemoryParams]):
    strategy_id = "gen_local_1790992235"

    @classmethod
    def params_type(cls) -> type[MemoryParams]:
        return MemoryParams

    def warmup_bars(self, params: MemoryParams) -> int:
        return 85

    def indicators(self, data: pd.DataFrame, params: MemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.empty(24, dtype=float)
        weights[0] = 1.0
        for lag in range(1, 24):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional_close = close * weights[0]
        for lag in range(1, 24):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional_close
        out["fractional_mean"] = fractional_close.rolling(20, min_periods=20).mean()
        out["fractional_std"] = fractional_close.rolling(20, min_periods=20).std(ddof=0)
        denominator = out["fractional_std"].where(out["fractional_std"] > 1e-12)
        out["shock"] = (fractional_close - out["fractional_mean"]) / denominator
        out["lag_one"] = out["shock"].shift(1)
        out["lag_two"] = out["shock"].shift(2)
        # Estimate memory through the previous close, excluding today's shock.
        out["corr_01"] = out["shock"].rolling(40, min_periods=40).corr(out["lag_one"]).shift(1)
        out["corr_02"] = out["shock"].rolling(40, min_periods=40).corr(out["lag_two"]).shift(1)
        out["corr_12"] = out["lag_one"].rolling(40, min_periods=40).corr(out["lag_two"]).shift(1)
        r01 = out["corr_01"].clip(-1.0, 1.0)
        r02 = out["corr_02"].clip(-1.0, 1.0)
        r12 = out["corr_12"].clip(-1.0, 1.0)
        out["partial_denominator"] = np.sqrt((1.0 - r02 ** 2) * (1.0 - r12 ** 2))
        safe_denominator = out["partial_denominator"].where(out["partial_denominator"] > 1e-8)
        out["partial_memory"] = ((r01 - r02 * r12) / safe_denominator).clip(-1.0, 1.0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["forecast"] = out["partial_memory"] * out["shock"]
        eligible = (
            np.isfinite(out["forecast"])
            & np.isfinite(out["atr"])
            & (out["atr"] > 0.0)
            & (out["partial_memory"].abs() >= 0.2)
            & (out["shock"].abs() >= 0.75)
        )
        out["candidate"] = np.where(eligible, np.sign(out["forecast"]), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MemoryParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        held_sessions = 0
        fixed_stop = np.nan
        for i in range(len(data)):
            if position != 0:
                if held_sessions == 0:
                    # Yesterday's decision fills at today's open. Freeze its ATR.
                    entry_price = opens[i]
                    entry_atr = atr[i - 1] if i > 0 else np.nan
                    if not (np.isfinite(entry_price) and entry_price > 0.0
                            and np.isfinite(entry_atr) and entry_atr > 0.0):
                        position = 0
                        decisions[i] = 0
                        continue
                    fixed_stop = entry_price - position * params.stop_atr * entry_atr
                held_sessions += 1
                stop_fired = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < fixed_stop)
                    or (position == -1 and closes[i] > fixed_stop)
                )
                if stop_fired or held_sessions >= 2 or not np.isfinite(closes[i]):
                    position = 0
                    held_sessions = 0
                    fixed_stop = np.nan
            elif i >= self.warmup_bars(params) - 1 and candidates[i] != 0:
                position = int(candidates[i])
                held_sessions = 0
            decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
