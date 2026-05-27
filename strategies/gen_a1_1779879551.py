from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DoublyStillParams:
    still_threshold: float = 0.2
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[DoublyStillParams]):
    strategy_id = "gen_a1_1779879551"

    @classmethod
    def params_type(cls):
        return DoublyStillParams

    def warmup_bars(self, params):
        return 300

    def indicators(self, data, params):
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(14, min_periods=14).mean()
        vov = atr.rolling(21, min_periods=21).std()
        atr_rank = atr.rolling(252, min_periods=252).rank(pct=True)
        vov_rank = vov.rolling(252, min_periods=252).rank(pct=True)

        return pd.DataFrame(
            {
                "atr": atr,
                "atr_rank": atr_rank,
                "vov_rank": vov_rank,
            },
            index=data.index,
        )

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        atr_rank = indicators["atr_rank"].to_numpy(dtype=float)
        vov_rank = indicators["vov_rank"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=np.int64)

        thresh = float(params.still_threshold)
        k = float(params.atr_mult)

        # NaN < x is False in numpy, so this mask is safe during warmup.
        still = (atr_rank < thresh) & (vov_rank < thresh)

        in_pos = False
        peak = -np.inf
        entry_atr = np.nan

        for i in range(n):
            if not in_pos:
                if still[i] and not np.isnan(atr[i]) and atr[i] > 0.0:
                    in_pos = True
                    peak = close[i]
                    entry_atr = atr[i]
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                if close[i] > peak:
                    peak = close[i]
                stop_level = peak - k * entry_atr
                if close[i] < stop_level:
                    in_pos = False
                    peak = -np.inf
                    entry_atr = np.nan
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(
            {
                "signal": signal.astype(np.int64),
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
