from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SessionCompressionParams:
    fast_window: int = 8
    slow_window: int = 60
    session_window: int = 12
    compression_threshold: float = 0.70
    overnight_ratio_threshold: float = 0.55
    efficiency_threshold: float = 0.30
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SessionCompressionParams]):
    strategy_id = "gen_local-w2_1790989782"

    @classmethod
    def params_type(cls) -> type[SessionCompressionParams]:
        return SessionCompressionParams

    def warmup_bars(self, params: SessionCompressionParams) -> int:
        return int(max(params.fast_window, params.slow_window, params.session_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SessionCompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["close_return"] = close.div(close.shift(1)).sub(1)
        out["overnight_return"] = opening.div(close.shift(1)).sub(1)
        out["session_return"] = close.div(opening).sub(1)
        out["fast_volatility"] = out["close_return"].rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["slow_volatility"] = out["close_return"].rolling(params.slow_window, min_periods=params.slow_window).std(ddof=0)
        out["overnight_volatility"] = out["overnight_return"].rolling(params.session_window, min_periods=params.session_window).std(ddof=0)
        out["session_volatility"] = out["session_return"].rolling(params.session_window, min_periods=params.session_window).std(ddof=0)
        out["session_drift"] = out["session_return"].rolling(params.session_window, min_periods=params.session_window).sum()
        out["session_travel"] = out["session_return"].abs().rolling(params.session_window, min_periods=params.session_window).sum()
        out["compression_ratio"] = out["fast_volatility"].div(out["slow_volatility"].where(out["slow_volatility"] > 0))
        out["overnight_ratio"] = out["overnight_volatility"].div(out["session_volatility"].where(out["session_volatility"] > 0))
        out["session_efficiency"] = out["session_drift"].div(out["session_travel"].where(out["session_travel"] > 0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SessionCompressionParams) -> SignalFrame:
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params) - 1, index=data.index)
        eligible = ready & indicators["compression_ratio"].le(params.compression_threshold) & indicators["overnight_ratio"].le(params.overnight_ratio_threshold)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[eligible & indicators["session_efficiency"].ge(params.efficiency_threshold)] = 1
        candidate.loc[eligible & indicators["session_efficiency"].le(-params.efficiency_threshold)] = -1
        previous = candidate.shift(1).fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(previous) & candidate.ne(0), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        holding_bars = max(1, int(params.holding_bars))
        for i, direction in enumerate(confirmed.to_numpy(dtype=int)):
            if position != 0:
                if i - entry_bar >= holding_bars:
                    position = 0
            elif direction != 0:
                position = int(direction)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
