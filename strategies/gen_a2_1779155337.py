from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# --- fixed structural constants (not tunable; the twist caps tunable params at 2) ---
_VR_Q = 5            # aggregation horizon for the variance-ratio statistic
_ATR_WINDOW = 14     # true-range smoothing window for the trailing stop
_VR_THRESHOLD = 1.0  # VR > 1 => positive serial correlation (persistent-trend regime)


@dataclass(slots=True)
class GeneratedParams:
    vr_window: int = 60
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779155337"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.vr_window) + _VR_Q + _ATR_WINDOW + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        w = max(int(params.vr_window), 2)

        # close-to-close returns: one-bar and overlapping q-bar
        r1 = close.pct_change()
        rq = close.pct_change(_VR_Q)

        var1 = r1.rolling(w).var()
        varq = rq.rolling(w).var()

        # Lo-MacKinlay variance ratio: > 1 => positive autocorrelation
        vr = varq / (_VR_Q * var1)
        vr = vr.replace([np.inf, -np.inf], np.nan)

        # ATR for the rolling-high trailing stop
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_WINDOW).mean()

        out = pd.DataFrame(index=data.index)
        out["vr"] = vr
        out["ret"] = r1
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        idx = data.index
        close = data["close"].to_numpy(dtype=float)
        vr = indicators["vr"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        k = float(params.atr_mult)
        thr = _VR_THRESHOLD

        in_pos = False
        hwm = 0.0  # highest close since entry (ratchets up only)

        for i in range(1, n):
            a = atr[i]
            if in_pos:
                if close[i] > hwm:
                    hwm = close[i]
                if np.isfinite(a) and a > 0.0:
                    stop = hwm - k * a
                    if close[i] < stop:
                        in_pos = False
                        raw[i] = 0
                        continue
                raw[i] = 1
            else:
                cross = (
                    np.isfinite(vr[i])
                    and np.isfinite(vr[i - 1])
                    and vr[i] > thr
                    and vr[i - 1] <= thr
                )
                rising = np.isfinite(ret[i]) and ret[i] > 0.0
                tradeable = np.isfinite(a) and a > 0.0
                if cross and rising and tradeable:
                    in_pos = True
                    hwm = close[i]
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=idx)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
