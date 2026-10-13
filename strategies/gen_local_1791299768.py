from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    range_window: int = 40
    streak_window: int = 126
    compression_percentile: float = 0.35
    streak_percentile: float = 0.85
    atr_period: int = 14
    stop_atr: float = 2.0
    max_hold: int = 10
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791299768"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.range_window + params.streak_window + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        compressed = true_range.lt(
            true_range.rolling(params.range_window).quantile(
                params.compression_percentile
            ).shift(1)
        )
        streak = compressed.astype(int).groupby((~compressed).cumsum()).cumsum()
        out = pd.DataFrame(index=data.index)
        out["compressed"] = compressed
        out["prior_streak"] = streak.shift(1).fillna(0).astype(int)
        out["streak_cutoff"] = streak.rolling(params.streak_window).quantile(
            params.streak_percentile
        ).shift(1)
        out["atr"] = true_range.rolling(params.atr_period).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy()
        lows = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        compressed = indicators["compressed"].to_numpy()
        prior_streak = indicators["prior_streak"].to_numpy()
        cutoff = indicators["streak_cutoff"].to_numpy()
        atr = indicators["atr"].to_numpy()
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        next_entry_bar = 0
        entry_atr = 0.0
        stop = np.nan

        for i in range(1, len(data)):
            if active:
                if i == entry_bar:
                    stop = opens[i] - params.stop_atr * entry_atr
                if i >= entry_bar and (
                    closes[i] < stop or i - entry_bar + 1 >= params.max_hold
                ):
                    active = False
                    next_entry_bar = i + 2
                else:
                    signal[i] = 1
            elif (
                i >= max(self.warmup_bars(params), next_entry_bar)
                and compressed[i - 1]
                and not compressed[i]
                and prior_streak[i] > 0
                and prior_streak[i] >= cutoff[i]
                and lows[i] < lows[i - 1]
                and closes[i] > closes[i - 1]
                and np.isfinite(atr[i])
            ):
                active = True
                entry_bar = i + 2
                entry_atr = atr[i]
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
