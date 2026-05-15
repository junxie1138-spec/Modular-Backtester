from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    zscore_entry: float = 1.5
    atr_stop_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778884617"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 35

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=1)
        zscore = (close - ma) / std

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        ind = pd.DataFrame(index=data.index)
        ind["zscore"] = zscore
        ind["atr"] = atr
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        zscore = indicators["zscore"].to_numpy()
        atr = indicators["atr"].to_numpy()

        n = len(close)
        sig = np.zeros(n, dtype=np.int64)

        SPIKE_THRESH = 3.0
        REFRAC_BARS = 5

        in_trade = False
        high_water = 0.0
        last_spike_bar = -(REFRAC_BARS + 2)

        for i in range(n):
            z = zscore[i]
            c = close[i]
            a = atr[i]

            if np.isnan(z) or np.isnan(a):
                continue

            # Track when an extreme z-score spike fires (inside or outside a trade)
            if z > SPIKE_THRESH:
                last_spike_bar = i

            if in_trade:
                if c > high_water:
                    high_water = c
                stop = high_water - params.atr_stop_mult * a
                if c < stop:
                    sig[i] = 0
                    in_trade = False
                else:
                    sig[i] = 1
            else:
                in_refractory = (i - last_spike_bar) <= REFRAC_BARS
                if z > params.zscore_entry and not in_refractory:
                    sig[i] = 1
                    in_trade = True
                    high_water = c

        df = pd.DataFrame({"signal": sig, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
