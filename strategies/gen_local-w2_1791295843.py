from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_days: int = 3
    percentile_window: int = 60
    upper_pct: float = 0.85
    trend_days: int = 15
    atr_days: int = 14
    breakeven_pct: float = 0.012
    atr_mult: float = 1.6
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791295843"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.percentile_window + params.roc_days + 2, params.trend_days + 1, params.atr_days)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        acceleration = close.pct_change(params.roc_days).diff()
        history = acceleration.shift(1)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["acceleration"] = acceleration.fillna(0.0)
        out["prior_min"] = history.rolling(3).min().fillna(np.inf)
        out["upper"] = history.rolling(params.percentile_window).quantile(params.upper_pct).fillna(np.inf)
        out["lower"] = history.rolling(params.percentile_window).quantile(1.0 - params.upper_pct).fillna(-np.inf)
        out["trend"] = close.pct_change(params.trend_days).fillna(0.0)
        out["atr"] = true_range.rolling(params.atr_days).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        enter = (
            (indicators["trend"] > 0)
            & (indicators["prior_min"] < indicators["lower"])
            & (indicators["acceleration"] > indicators["upper"])
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        target = np.zeros(len(data), dtype=int)
        holding = False
        armed = False
        entry_price = 0.0
        stop = -np.inf
        age = 0

        for i in range(len(data)):
            if i and target[i - 1] and not holding:
                holding = True
                armed = False
                entry_price = float(opens[i])
                stop = -np.inf
                age = 0
            elif i and not target[i - 1] and holding:
                holding = False

            if holding:
                age += 1
                close = float(closes[i])
                if (armed and close <= stop) or age >= params.max_hold:
                    target[i] = 0
                    continue
                if close >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, close - params.atr_mult * float(atr[i]))
                target[i] = 1
            elif enter[i]:
                target[i] = 1

        df = pd.DataFrame({"signal": target, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
