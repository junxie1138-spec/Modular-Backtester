from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 10
    comp_lookback: int = 60
    comp_pct: float = 0.30
    roc_window: int = 5
    profit_target: float = 0.02
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779154623"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.comp_lookback + params.range_window + params.roc_window + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        W = max(2, int(params.range_window))
        L = max(5, int(params.comp_lookback))
        k = max(1, int(params.roc_window))
        q = float(params.comp_pct)
        if q < 0.01:
            q = 0.01
        if q > 0.99:
            q = 0.99

        # Normalised rolling range -> coiled-spring tension gauge.
        safe_close = close.replace(0.0, np.nan)
        rng = (high.rolling(W).max() - low.rolling(W).min()) / safe_close
        out["rng"] = rng

        # Compression primitive: range sits in its bottom quantile over L bars.
        thresh = rng.rolling(L).quantile(q)
        out["comp_thresh"] = thresh
        out["compressed"] = (rng <= thresh).fillna(False)

        # Rate-of-change and its acceleration (second difference of return).
        roc = close.pct_change(k)
        accel = roc.diff()
        out["roc"] = roc
        out["accel"] = accel
        out["accel_prev"] = accel.shift(1)

        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        df = pd.DataFrame(index=data.index)

        close = data["close"].to_numpy(dtype=float)

        compressed = indicators["compressed"].fillna(False).to_numpy(dtype=bool)
        accel = indicators["accel"].fillna(0.0).to_numpy(dtype=float)
        accel_prev = indicators["accel_prev"].fillna(0.0).to_numpy(dtype=float)

        # Two-primitive AND: range compressed AND fresh upward ROC-acceleration flip.
        accel_flip = (accel > 0.0) & (accel_prev <= 0.0)
        entry = compressed & accel_flip

        pt = float(params.profit_target)
        if pt <= 0.0:
            pt = 0.001
        max_hold = max(1, int(params.max_hold_bars))

        sig = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if pos == 0:
                if bool(entry[i]):
                    pos = 1
                    entry_price = close[i]
                    bars_held = 0
                    sig[i] = 1
                else:
                    sig[i] = 0
            else:
                bars_held += 1
                hit_target = entry_price > 0.0 and close[i] >= entry_price * (1.0 + pt)
                time_stop = bars_held >= max_hold
                if hit_target or time_stop:
                    pos = 0
                    entry_price = 0.0
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = 1

        signal = pd.Series(sig, index=data.index)
        signal = signal.shift(1).fillna(0).astype(int)

        df["signal"] = signal
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
