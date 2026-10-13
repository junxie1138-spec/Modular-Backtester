from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    profit_target: float = 0.02
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local_1790870336"

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return 100

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        previous_close = close.shift(1).where(lambda x: x > 0)
        weights = np.ones(40, dtype=float)
        for k in range(1, 40):
            weights[k] = -weights[k - 1] * (0.4 - k + 1) / k
        fractional_close = close.copy()
        for k in range(1, 40):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        memory_mean = fractional_close.rolling(60, min_periods=60).mean()
        memory_std = fractional_close.rolling(60, min_periods=60).std(ddof=0)
        memory_z = (fractional_close - memory_mean) / memory_std.where(memory_std > 0)
        returns = close / previous_close - 1.0
        volatility = returns.rolling(20, min_periods=20).std(ddof=0)
        gap = data["open"] / previous_close - 1.0
        result = pd.DataFrame(index=data.index)
        result["previous_close"] = previous_close
        result["fractional_close"] = fractional_close
        result["memory_mean"] = memory_mean
        result["memory_std"] = memory_std
        result["memory_z"] = memory_z
        result["previous_memory_z"] = memory_z.shift(1)
        result["returns"] = returns
        result["volatility"] = volatility
        result["gap"] = gap
        result["entry"] = (
            (gap < -0.25 * volatility)
            & (volatility > 0)
            & (close > previous_close)
            & (memory_z > 0)
            & (memory_z.shift(1) <= 0)
        ).fillna(False)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        holding = False
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if holding:
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        holding = False
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_expired:
                    holding = False
                    entry_price = np.nan
                    entry_bar = -1
                else:
                    desired[i] = 1
            elif entries[i] and i >= self.warmup_bars(params) - 1:
                holding = True
                entry_bar = -1
                desired[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        assert df["signal"].isin([0, 1]).all() and (df["size"] > 0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
