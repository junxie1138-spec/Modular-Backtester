from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_streak: int = 3
    streak_mult: float = 1.3
    baseline_window: int = 60
    confirm_bars: int = 2
    atr_window: int = 14
    k_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    """Selling-shockwave dissipation: buy after an abnormally long down-close
    streak ends and two up-closes confirm the recovery wave has started."""

    strategy_id = "gen_a1_1778903886"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.baseline_window, params.atr_window)) + 5

    def indicators(self, data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]

        diff = close.diff()
        dn = (diff < 0).astype("int64")
        up = (diff > 0).astype("int64")

        # Consecutive run lengths of down-closes and up-closes.
        dn_grp = (dn != dn.shift()).cumsum()
        down_streak = dn * (dn.groupby(dn_grp).cumcount() + 1)

        up_grp = (up != up.shift()).cumsum()
        up_streak = up * (up.groupby(up_grp).cumcount() + 1)

        # Length of the most recently *completed* down-streak, captured on the
        # first up-close that breaks it and forward-filled thereafter.
        prev_ds = down_streak.shift(1)
        ended = pd.Series(np.nan, index=close.index)
        mask = (up_streak == 1) & (prev_ds > 0)
        ended[mask] = prev_ds[mask].astype("float64")
        last_down_streak = ended.ffill()

        # Typical recent down-streak length - the shockwave baseline amplitude.
        baseline = last_down_streak.rolling(
            int(params.baseline_window), min_periods=5
        ).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(
            int(params.atr_window), min_periods=int(params.atr_window)
        ).mean()

        out = pd.DataFrame(index=data.index)
        out["up_streak"] = up_streak
        out["last_down_streak"] = last_down_streak
        out["baseline"] = baseline
        out["atr"] = atr
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype="float64")
        up_streak = indicators["up_streak"].to_numpy(dtype="float64")
        last_ds = indicators["last_down_streak"].to_numpy(dtype="float64")
        baseline = indicators["baseline"].to_numpy(dtype="float64")
        atr = indicators["atr"].to_numpy(dtype="float64")

        n = len(close)
        confirm = float(int(params.confirm_bars))

        # Two-bar confirmation twist: exactly `confirm` consecutive up-closes
        # following an abnormally long (shockwave) down-streak that has ended.
        # NaN comparisons evaluate False, so warmup bars never trigger entry.
        entry_cond = (
            (up_streak == confirm)
            & (last_ds >= float(params.min_streak))
            & (last_ds >= float(params.streak_mult) * baseline)
        )

        raw = np.zeros(n, dtype="int64")
        position = 0
        hwm = 0.0
        for i in range(n):
            if position == 0:
                if bool(entry_cond[i]):
                    position = 1
                    hwm = close[i]
            else:
                # Rolling-high trailing stop: ratchet the high-water mark up,
                # exit when close falls k*ATR below the in-trade high.
                if close[i] > hwm:
                    hwm = close[i]
                a = atr[i]
                if not np.isnan(a):
                    stop = hwm - float(params.k_atr) * a
                    if close[i] <= stop:
                        position = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
