from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityMemoryParams:
    graph_bars: int = 8
    memory_bars: int = 60
    percentile_bars: int = 126
    entry_percentile: float = 0.80
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityMemoryParams]):
    strategy_id = "gen_local-w2_1790993852"

    @classmethod
    def params_type(cls) -> type[VisibilityMemoryParams]:
        return VisibilityMemoryParams

    def warmup_bars(self, params: VisibilityMemoryParams) -> int:
        return int(params.memory_bars + params.graph_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityMemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        magnitude = returns.abs()
        out["return"] = returns
        out["magnitude"] = magnitude
        weighted_memory = pd.Series(0.0, index=data.index)
        weighted_degree = pd.Series(0.0, index=data.index)
        blocker = magnitude.shift(1)

        # Horizontal visibility: both endpoints exceed every intervening shock.
        # Exclude adjacent edges, which exist regardless of graph topology.
        for lag in range(2, params.graph_bars + 1):
            if lag > 2:
                blocker = pd.Series(
                    np.maximum(blocker.to_numpy(), magnitude.shift(lag - 1).to_numpy()),
                    index=data.index,
                )
            past_magnitude = magnitude.shift(lag)
            visible = (magnitude > blocker) & (past_magnitude > blocker)
            correlation = returns.rolling(
                params.memory_bars, min_periods=params.memory_bars
            ).corr(returns.shift(lag)).replace([np.inf, -np.inf], np.nan)
            weight = visible.astype(float) / float(lag)
            out[f"blocker_{lag}"] = blocker
            out[f"past_magnitude_{lag}"] = past_magnitude
            out[f"visible_{lag}"] = visible.astype(int)
            out[f"correlation_{lag}"] = correlation
            out[f"weight_{lag}"] = weight
            weighted_memory = weighted_memory + weight * correlation.fillna(0.0)
            weighted_degree = weighted_degree + weight

        memory = (weighted_memory / weighted_degree.replace(0.0, np.nan)).fillna(0.0)
        ready = returns.notna() & returns.shift(params.memory_bars + params.graph_bars).notna()
        forecast = (returns * memory).where(ready).replace([np.inf, -np.inf], np.nan)
        threshold = forecast.abs().shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        qualifying = forecast.notna() & threshold.notna() & (forecast.abs() > threshold)
        out["weighted_memory"] = weighted_memory
        out["weighted_degree"] = weighted_degree
        out["memory"] = memory
        out["ready"] = ready.astype(int)
        out["forecast"] = forecast
        out["entry_threshold"] = threshold
        out["entry"] = np.where(qualifying, np.sign(forecast.fillna(0.0)), 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityMemoryParams,
    ) -> SignalFrame:
        candidates = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        pending = 0
        entry_price = 0.0
        entered_bar = -1

        for i in range(len(data)):
            # The preceding close's entry decision fills at this bar's open.
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    side = pending
                    entry_price = opens[i]
                    entered_bar = i
                pending = 0

            if side:
                gain = side * (closes[i] / entry_price - 1.0)
                profit_exit = np.isfinite(gain) and gain >= params.profit_target
                time_exit = i - entered_bar + 1 >= params.hold_bars
                if profit_exit or time_exit:
                    side = 0
                raw[i] = side
            elif candidates[i]:
                pending = int(candidates[i])
                raw[i] = pending

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
