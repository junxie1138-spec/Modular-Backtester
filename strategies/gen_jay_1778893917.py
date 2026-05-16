from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_streak: int = 3
    acorr_window: int = 30
    profit_target_pct: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_jay_1778893917'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # acorr needs acorr_window + 2 bars (pct_change + shift(1) + rolling)
        return params.acorr_window + params.min_streak + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        ret = data['close'].pct_change()

        # Consecutive down-close streak (queue depth)
        down_day = (data['close'] < data['close'].shift(1)).astype(int)
        grp = (down_day != down_day.shift(1)).cumsum()
        streak = down_day.groupby(grp).cumcount() + 1
        ind['down_streak'] = streak.where(down_day == 1, 0).astype(float)

        # Rolling lag-1 return autocorrelation (mean-reversion regime detector)
        ind['acorr'] = ret.rolling(params.acorr_window).corr(ret.shift(1))

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = data.copy()
        close = data['close'].values
        streak_v = indicators['down_streak'].values
        acorr_v = indicators['acorr'].values
        n = len(df)

        raw_signal = np.zeros(n, dtype=int)
        pt = params.profit_target_pct / 100.0
        in_trade = False
        entry_bar = -1

        for i in range(n):
            if in_trade:
                bars_held = i - entry_bar
                pnl = (close[i] - close[entry_bar]) / close[entry_bar]
                if bars_held < params.max_hold_bars and pnl < pt:
                    raw_signal[i] = 1
                else:
                    in_trade = False
            else:
                s = streak_v[i]
                ac = acorr_v[i]
                if (
                    not np.isnan(s)
                    and not np.isnan(ac)
                    and s >= params.min_streak
                    and ac < 0.0
                ):
                    raw_signal[i] = 1
                    in_trade = True
                    entry_bar = i

        df['signal'] = (
            pd.Series(raw_signal, index=df.index)
            .shift(1)
            .fillna(0)
            .astype(int)
        )
        df['size'] = 1.0

        return SignalFrame(data=df, signal_column='signal', size_column='size')
