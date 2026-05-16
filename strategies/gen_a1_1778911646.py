from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakVolParams:
    streak_window: int = 252
    streak_pct: float = 0.90
    vol_window: int = 20
    target_vol: float = 0.01
    size_min: float = 0.2
    size_max: float = 1.0
    min_streak: int = 2


class GeneratedStrategy(BaseStrategy[StreakVolParams]):
    strategy_id = "gen_a1_1778911646"

    @classmethod
    def params_type(cls):
        return StreakVolParams

    def warmup_bars(self, params):
        return int(max(params.streak_window, params.vol_window) + 2)

    def indicators(self, data, params):
        close = data["close"]
        ret = close.pct_change()

        # Sign of each daily return: +1 up day, -1 down day, 0 flat.
        sign = np.sign(ret).fillna(0.0)

        # Consecutive-streak count: length of the current same-sign run.
        grp = (sign != sign.shift()).cumsum()
        run_len = sign.groupby(grp).cumcount().add(1).astype(float)

        # Percentile threshold (adaptive yield point) instead of a fixed level.
        thr = run_len.rolling(
            params.streak_window, min_periods=params.streak_window
        ).quantile(params.streak_pct)

        # A run is 'plastic' once it exceeds its own rolling percentile.
        # NaN thr during warmup -> comparison is False -> no signal.
        is_extreme = (run_len > thr) & (run_len >= float(params.min_streak))
        extreme_up = (is_extreme & (sign > 0)).astype(float)
        extreme_down = (is_extreme & (sign < 0)).astype(float)

        # Volatility-targeted position size.
        rvol = ret.rolling(params.vol_window, min_periods=params.vol_window).std()
        size = (params.target_vol / rvol.replace(0.0, np.nan)).clip(
            lower=params.size_min, upper=params.size_max
        )
        size = size.fillna(params.size_min)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["sign"] = sign
        out["run_len"] = run_len
        out["thr"] = thr
        out["rvol"] = rvol
        out["size"] = size
        out["extreme_up"] = extreme_up
        out["extreme_down"] = extreme_down
        return out

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        sign_arr = indicators["sign"].to_numpy()
        eu = indicators["extreme_up"].to_numpy() > 0.5
        ed = indicators["extreme_down"].to_numpy() > 0.5

        # Signal-reversal exit: enter on an extreme streak, hold while the run
        # continues, exit (or flip) only when the return sign flips against the
        # position - the entry condition has reversed.
        sig = np.zeros(n, dtype=np.int64)
        pos = 0
        for i in range(n):
            if pos == 0:
                if eu[i]:
                    pos = 1
                elif ed[i]:
                    pos = -1
            elif pos == 1:
                if sign_arr[i] < 0.0:
                    pos = -1 if ed[i] else 0
            else:  # pos == -1
                if sign_arr[i] > 0.0:
                    pos = 1 if eu[i] else 0
            sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].shift(1).fillna(params.size_min)
        size = size.clip(lower=params.size_min).astype(float)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
