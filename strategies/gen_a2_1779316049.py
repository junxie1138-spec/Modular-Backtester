from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_threshold: int = 4
    profit_target_pct: float = 0.04
    time_stop_bars: int = 17
    require_up_direction: bool = True


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779316049"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # pct_change (1) + sign-of-prev (1) + one bar to observe the break = 3
        return 3

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()
        sgn = np.sign(ret.fillna(0.0))
        prev_sgn = sgn.shift(1).fillna(0.0)
        # 1 iff today's return sign matches yesterday's AND today is non-flat
        agree = ((sgn == prev_sgn) & (sgn != 0)).astype(int)

        # Consecutive-streak count of sign-agreement (vectorised run-length).
        # Every break starts a new group; cumsum of agree inside the group is
        # the current streak length at each bar.
        breaks = (agree == 0).astype(int)
        grp = breaks.cumsum()
        streak = agree.groupby(grp).cumsum().astype(int)

        ind = pd.DataFrame(
            {
                "ret": ret,
                "sgn": sgn,
                "agree": agree,
                "streak": streak,
            },
            index=data.index,
        )
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].fillna(0).to_numpy(dtype=np.int64)
        sgn = indicators["sgn"].fillna(0.0).to_numpy(dtype=float)

        n = len(data)
        signal = np.zeros(n, dtype=np.int64)

        threshold = int(params.streak_threshold)
        prof = float(params.profit_target_pct)
        time_stop = int(params.time_stop_bars)
        require_up = bool(params.require_up_direction)

        in_pos = False
        entry_idx = -1
        entry_price = 0.0

        for i in range(n):
            if in_pos:
                bars_held = i - entry_idx
                if entry_price > 0.0 and np.isfinite(close[i]):
                    move = close[i] / entry_price - 1.0
                else:
                    move = 0.0
                if (not np.isfinite(move)) or move >= prof or bars_held >= time_stop:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                # Shockwave release: yesterday the streak was packed (>=K) and
                # the packed direction was up, today the streak collapsed to 0.
                if i >= 1:
                    prior_streak = streak[i - 1]
                    prior_sgn = sgn[i - 1]
                    cur_streak = streak[i]
                    cur_close = close[i]
                    if (
                        prior_streak >= threshold
                        and cur_streak == 0
                        and ((not require_up) or prior_sgn > 0.0)
                        and np.isfinite(cur_close)
                        and cur_close > 0.0
                    ):
                        in_pos = True
                        entry_idx = i
                        entry_price = cur_close
                        signal[i] = 1

        df = pd.DataFrame(
            {
                "signal": signal,
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )
        # Mandatory one-bar shift: decision on bar N's close fills on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
