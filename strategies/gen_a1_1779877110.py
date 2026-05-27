from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 10
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779877110"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params):
        w = int(params.window)
        return max(w, 14) + 2

    @classmethod
    def indicators(cls, data, params):
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

        bar_range = high - low
        w = int(params.window)
        prior_max_range = bar_range.shift(1).rolling(w, min_periods=w).max()
        n_ago_close = close.shift(w)

        out = pd.DataFrame(
            {
                "atr": atr,
                "bar_range": bar_range,
                "prior_max_range": prior_max_range,
                "n_ago_close": n_ago_close,
            },
            index=data.index,
        )
        return out

    @classmethod
    def generate_signals(cls, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        bar_range = indicators["bar_range"].to_numpy(dtype=float)
        prior_max_range = indicators["prior_max_range"].to_numpy(dtype=float)
        n_ago_close = indicators["n_ago_close"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)
        k = float(params.atr_mult)

        pos = 0
        hwm = 0.0
        lwm = 0.0

        for i in range(n):
            c = close[i]
            a = atr[i]
            br = bar_range[i]
            pmr = prior_max_range[i]
            nac = n_ago_close[i]

            if pos == 0:
                ready = (
                    not np.isnan(a)
                    and not np.isnan(br)
                    and not np.isnan(pmr)
                    and not np.isnan(nac)
                    and a > 0.0
                )
                if ready and br > pmr:
                    if c > nac:
                        pos = 1
                        hwm = c
                    elif c < nac:
                        pos = -1
                        lwm = c
                raw[i] = pos
            elif pos == 1:
                if not np.isnan(c) and c > hwm:
                    hwm = c
                if not np.isnan(a) and not np.isnan(c) and c < hwm - k * a:
                    pos = 0
                raw[i] = pos
            else:
                if not np.isnan(c) and c < lwm:
                    lwm = c
                if not np.isnan(a) and not np.isnan(c) and c > lwm + k * a:
                    pos = 0
                raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
