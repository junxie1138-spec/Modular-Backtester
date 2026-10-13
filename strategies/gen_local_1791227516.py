from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPhaseParams:
    history: int = 126
    percentile: float = 0.8
    target_pct: float = 0.02
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalPhaseParams]):
    strategy_id = "gen_local_1791227516"

    @classmethod
    def params_type(cls):
        return FractionalPhaseParams

    def warmup_bars(self, params: FractionalPhaseParams) -> int:
        return 16 + 20 + params.history + 1

    def indicators(self, data: pd.DataFrame, params: FractionalPhaseParams) -> pd.DataFrame:
        close = data["close"].replace(0, np.nan)

        def fractional(order: float) -> pd.Series:
            weights = [1.0]
            for lag in range(1, 16):
                weights.append(weights[-1] * (lag - 1 - order) / lag)
            return sum(weight * close.shift(lag) for lag, weight in enumerate(weights))

        fast_raw = fractional(0.8)
        slow_raw = fractional(0.35)
        fast = (fast_raw - fast_raw.rolling(20).mean()) / close
        slow = (slow_raw - slow_raw.rolling(20).mean()) / close
        prior = fast.shift(1).rolling(params.history, min_periods=params.history)
        out = pd.DataFrame(index=data.index)
        out["fast"] = fast
        out["slow"] = slow
        out["upper"] = prior.quantile(params.percentile)
        out["lower"] = prior.quantile(1.0 - params.percentile)
        out["ready"] = out.notna().all(axis=1)
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalPhaseParams,
    ) -> SignalFrame:
        ready = indicators["ready"] & indicators["ready"].shift(1, fill_value=False)
        fast = indicators["fast"]
        upper = indicators["upper"]
        lower = indicators["lower"]
        long_entry = (ready & (fast > upper) & (fast.shift(1) <= upper.shift(1)) & (indicators["slow"] > 0)).to_numpy()
        short_entry = (ready & (fast < lower) & (fast.shift(1) >= lower.shift(1)) & (indicators["slow"] < 0)).to_numpy()

        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        direction = 0
        held = 0
        entry_price = 0.0
        for i in range(len(data)):
            if direction == 0:
                if long_entry[i] or short_entry[i]:
                    direction = 1 if long_entry[i] else -1
                    signal[i] = direction
                    held = 0
            else:
                if held == 0:
                    entry_price = opens[i]
                held += 1
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.target_pct or held >= params.hold_bars:
                    direction = 0
                    held = 0
                else:
                    signal[i] = direction

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
