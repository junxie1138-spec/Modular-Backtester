from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 12
    peak_window: int = 20
    min_drawdown: float = 0.02
    max_drawdown: float = 0.08
    profit_target: float = 0.02
    time_stop_bars: int = 3


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791226246"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.peak_window, params.fractional_window + 5)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        ma200 = close.rolling(200, min_periods=200).mean()
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()

        fractional = close * 0.0
        weight = 1.0
        for lag in range(params.fractional_window):
            if lag:
                weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        smooth = fractional.rolling(5, min_periods=5).mean()

        out["trend_ok"] = (close > ma200).fillna(False)
        out["drawdown"] = (1.0 - close / peak.replace(0, np.nan)).clip(lower=0).fillna(0.0)
        out["fractional_gap"] = (fractional - smooth).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        gap = indicators["fractional_gap"]
        entry = (
            indicators["trend_ok"]
            & indicators["drawdown"].between(params.min_drawdown, params.max_drawdown)
            & gap.gt(0)
            & gap.shift(1).le(0).fillna(False)
        ).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = 0.0
        flat_until = -1

        for i in range(len(data)):
            if active:
                signal[i] = 1
                if i == entry_bar:
                    entry_price = opens[i]
                if i >= entry_bar and (
                    closes[i] >= entry_price * (1.0 + params.profit_target)
                    or i - entry_bar + 1 >= params.time_stop_bars
                ):
                    signal[i] = 0
                    active = False
                    flat_until = i + 1
            elif i >= self.warmup_bars(params) and i > flat_until and i + 2 < len(data) and entry[i]:
                active = True
                entry_bar = i + 2
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
