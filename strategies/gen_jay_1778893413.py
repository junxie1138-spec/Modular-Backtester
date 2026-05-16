from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    profit_target_pct: float = 0.5
    time_stop_bars: int = 2


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778893413"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.streak_len * 5 + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        daily_ret = data["close"].pct_change()
        # NaN > 0 evaluates to False in pandas boolean context
        is_positive = daily_ret > 0

        streak = pd.Series(0.0, index=data.index)

        for wd in range(5):
            wd_mask = data.index.dayofweek == wd
            if wd_mask.sum() == 0:
                continue
            wd_pos = is_positive.loc[wd_mask].reset_index(drop=True)
            # Vectorised consecutive run count within each weekday's own time series
            shifted = wd_pos.shift(1).fillna(False).astype(bool)
            changed = (wd_pos != shifted).cumsum()
            wd_streak = wd_pos.groupby(changed).cumcount() + 1
            wd_streak = wd_streak.where(wd_pos, 0)
            streak.loc[wd_mask] = wd_streak.values

        ind = pd.DataFrame(index=data.index)
        ind["streak"] = streak
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        streak = indicators["streak"]

        raw_entry = (streak >= params.streak_len).astype(int)
        # Mandatory 1-bar shift: decide on bar N close, fill on bar N+1
        raw_entry = raw_entry.shift(1).fillna(0).astype(int)

        sig_arr = raw_entry.values.copy()
        size_arr = np.ones(len(sig_arr))

        open_arr = data["open"].values
        close_arr = data["close"].values
        wd_arr = data.index.dayofweek.values

        in_trade = False
        entry_price = 0.0
        bars_held = 0
        # Per-weekday refractory: blocks re-entry for streak_len same-weekday bars after exit
        wd_refractory: dict[int, int] = {}

        for i in range(len(sig_arr)):
            wd = int(wd_arr[i])

            if wd in wd_refractory:
                wd_refractory[wd] -= 1
                if wd_refractory[wd] <= 0:
                    del wd_refractory[wd]

            if in_trade:
                bars_held += 1
                ret_pct = (close_arr[i] / entry_price - 1.0) * 100.0
                if ret_pct >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    sig_arr[i] = 0
                    in_trade = False
                    wd_refractory[wd] = params.streak_len
                else:
                    sig_arr[i] = 1
            elif sig_arr[i] == 1:
                if wd in wd_refractory:
                    sig_arr[i] = 0
                else:
                    in_trade = True
                    entry_price = open_arr[i]
                    bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig_arr.astype(int)
        df["size"] = size_arr

        return SignalFrame(data=df, signal_column="signal", size_column="size")
