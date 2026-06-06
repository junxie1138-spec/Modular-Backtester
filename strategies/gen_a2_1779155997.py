from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OvernightIntradayParams:
    window: int = 5
    entry_thresh: float = 0.005
    vol_window: int = 20
    target_vol: float = 0.02
    size_floor: float = 0.5
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[OvernightIntradayParams]):
    """Decompose each daily bar into an overnight gap return and an intraday
    session return. Go long only when rolling sums of BOTH components are
    strongly bullish (two-primitive AND). Hold via hysteresis: exit only when
    the entry agreement flips, i.e. one of the two components turns negative.
    """

    strategy_id = "gen_a2_1779155997"

    @classmethod
    def params_type(cls):
        return OvernightIntradayParams

    def warmup_bars(self, params):
        return int(max(int(params.window), int(params.vol_window))) + 2

    def indicators(self, data, params):
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        prev_close = close.shift(1)

        # Primitive 1: overnight gap return (open vs prior close).
        overnight = open_ / prev_close - 1.0
        # Primitive 2: intraday session return (close vs same-day open).
        intraday = close / open_ - 1.0

        overnight = overnight.replace([np.inf, -np.inf], np.nan)
        intraday = intraday.replace([np.inf, -np.inf], np.nan)

        w = max(int(params.window), 1)
        ov_sum = overnight.rolling(w, min_periods=w).sum()
        id_sum = intraday.rolling(w, min_periods=w).sum()

        vw = max(int(params.vol_window), 2)
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        realized_vol = ret.rolling(vw, min_periods=vw).std()

        # Inverse-vol position sizing, bounded and NaN-safe.
        size_raw = float(params.target_vol) / realized_vol.replace(0.0, np.nan)
        size_raw = size_raw.clip(lower=float(params.size_floor),
                                 upper=float(params.size_cap))
        size_raw = size_raw.fillna(float(params.size_floor))

        ind = pd.DataFrame(index=data.index)
        ind["overnight"] = overnight
        ind["intraday"] = intraday
        ind["ov_sum"] = ov_sum
        ind["id_sum"] = id_sum
        ind["realized_vol"] = realized_vol
        ind["size"] = size_raw
        return ind

    def generate_signals(self, data, indicators, ctx, params):
        ov = indicators["ov_sum"].to_numpy(dtype=float)
        idd = indicators["id_sum"].to_numpy(dtype=float)
        n = len(data)

        thr = float(params.entry_thresh)
        sig = np.zeros(n, dtype=np.int64)
        pos = 0
        for i in range(n):
            o = ov[i]
            d = idd[i]
            if not np.isfinite(o) or not np.isfinite(d):
                # Warmup / missing data: stay flat, reset state.
                pos = 0
                sig[i] = 0
                continue
            if pos == 0:
                # Entry: both overnight and intraday pressure strongly bullish.
                if o > thr and d > thr:
                    pos = 1
            else:
                # Signal-reversal exit: hold until the bullish AND-agreement
                # flips, i.e. one component drops back to non-positive.
                if o <= 0.0 or d <= 0.0:
                    pos = 0
            sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index)
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].astype(float)
        size = size.clip(lower=float(params.size_floor),
                         upper=float(params.size_cap))
        size = size.fillna(float(params.size_floor))
        size = size.where(size > 0.0, float(params.size_floor))
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
