from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    channel_bars: int = 20
    memory_bars: int = 18
    minimum_streak: int = 2

    def __post_init__(self):
        if self.channel_bars < 2 or self.memory_bars < 2:
            raise ValueError('Channel and memory windows must be at least two bars.')
        if not 2 <= self.minimum_streak <= 4:
            raise ValueError('minimum_streak must be between two and four.')


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = 'gen_local_1790968382'

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(params.channel_bars + params.memory_bars + 5)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['channel_high'] = close.shift(1).rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out['channel_low'] = close.shift(1).rolling(params.channel_bars, min_periods=params.channel_bars).min()
        out['valid'] = close.notna() & out['channel_high'].notna() & out['channel_low'].notna()
        out['up_breakout'] = out['valid'] & close.gt(out['channel_high'])
        out['down_breakout'] = out['valid'] & close.lt(out['channel_low'])
        for side in ('up', 'down'):
            breakout = out[side + '_breakout']
            # Four consecutive breakouts saturate the persistence score.
            out[side + '_streak'] = breakout.astype(int).groupby((~breakout).cumsum()).cumsum().clip(upper=4)
            out[side + '_peak'] = out[side + '_streak'].rolling(params.memory_bars, min_periods=params.memory_bars).max()
        out['ready'] = out['valid'].rolling(params.memory_bars, min_periods=params.memory_bars).sum().eq(params.memory_bars)
        out['eligible'] = (
            out['ready']
            & out['up_peak'].ge(params.minimum_streak)
            & out['up_peak'].gt(out['down_peak'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        eligible = indicators['eligible'].fillna(False).astype(bool)
        df = pd.DataFrame(index=data.index)
        # Two consecutive qualifying closes enter; the first false close exits.
        df['signal'] = (eligible & eligible.shift(1, fill_value=False)).astype(int)
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'eligible': [False, True, True, True, False, True, True]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, StreakParams()).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
