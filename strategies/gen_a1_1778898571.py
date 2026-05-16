from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TurnOfWeekStreakParams:
    entry_streak: int = 3
    profit_target: float = 0.02
    time_stop: int = 4
    refractory_bars: int = 3
    long_weekday: int = 0
    short_weekday: int = 4
    conviction_scale: float = 0.15


def _consecutive(mask: pd.Series) -> pd.Series:
    """Length of the current consecutive run of True values; 0 where False."""
    mask = mask.fillna(False).astype(bool)
    grp = (~mask).cumsum()
    run = mask.groupby(grp).cumcount() + 1
    return run.where(mask, 0).astype(int)


class GeneratedStrategy(BaseStrategy[TurnOfWeekStreakParams]):
    strategy_id = "gen_a1_1778898571"

    @classmethod
    def params_type(cls):
        return TurnOfWeekStreakParams

    @staticmethod
    def warmup_bars(params: TurnOfWeekStreakParams) -> int:
        # Only pct_change (1 bar) plus a small safety margin; streak runs need no window.
        return 5

    def indicators(self, data: pd.DataFrame, params: TurnOfWeekStreakParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        down_streak = _consecutive(ret < 0)
        up_streak = _consecutive(ret > 0)
        dow = pd.Series(np.asarray(data.index.dayofweek), index=data.index)

        out = pd.DataFrame(index=data.index)
        out["down_streak"] = down_streak
        out["up_streak"] = up_streak
        out["dayofweek"] = dow

        long_gate = (down_streak >= params.entry_streak) & (dow == params.long_weekday)
        short_gate = (up_streak >= params.entry_streak) & (dow == params.short_weekday)
        out["entry_long"] = long_gate.fillna(False).astype(int)
        out["entry_short"] = short_gate.fillna(False).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TurnOfWeekStreakParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_long = indicators["entry_long"].to_numpy()
        entry_short = indicators["entry_short"].to_numpy()
        down_streak = indicators["down_streak"].to_numpy()
        up_streak = indicators["up_streak"].to_numpy()

        pos = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        state = 0
        entry_price = 0.0
        bars_held = 0
        refractory = 0
        cur_size = 1.0

        time_stop = max(int(params.time_stop), 1)
        refractory_bars = max(int(params.refractory_bars), 0)

        for i in range(n):
            # --- exit logic: profit-target OR time-stop, whichever fires first ---
            if state != 0:
                bars_held += 1
                if entry_price > 0.0:
                    pnl = (close[i] / entry_price - 1.0) * state
                else:
                    pnl = 0.0
                if pnl >= params.profit_target or bars_held >= time_stop:
                    state = 0
                    refractory = refractory_bars
                    cur_size = 1.0

            # --- entry logic: blocked during the post-exit refractory period ---
            if state == 0:
                if refractory > 0:
                    refractory -= 1
                elif entry_long[i] == 1:
                    state = 1
                    entry_price = close[i]
                    bars_held = 0
                    excess = int(down_streak[i]) - params.entry_streak
                    cur_size = 1.0 + params.conviction_scale * max(excess, 0)
                elif entry_short[i] == 1:
                    state = -1
                    entry_price = close[i]
                    bars_held = 0
                    excess = int(up_streak[i]) - params.entry_streak
                    cur_size = 1.0 + params.conviction_scale * max(excess, 0)

            pos[i] = state
            size[i] = cur_size if state != 0 else 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = np.where(size > 0.0, size, 1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
