from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    streak_min: int = 3
    efficiency_min: float = 0.55
    hold_bars: int = 18

    def __post_init__(self):
        if not 1 <= self.streak_min <= 8:
            raise ValueError('streak_min must be between 1 and 8')
        if not 0.0 <= self.efficiency_min <= 1.0:
            raise ValueError('efficiency_min must be between 0 and 1')
        if not 15 <= self.hold_bars <= 20:
            raise ValueError('hold_bars must be between 15 and 20')


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = 'gen_local_1791251275'

    @classmethod
    def params_type(cls):
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['rising_low'] = data['low'].gt(data['low'].shift(1)).fillna(False)
        alive = pd.Series(True, index=data.index)
        streak = pd.Series(0, index=data.index, dtype=int)
        for lag in range(8):
            alive = alive & out['rising_low'].shift(lag, fill_value=False)
            streak = streak + alive.astype(int)
        out['low_streak'] = streak
        out['prior_ceiling'] = data['high'].shift(1).rolling(8, min_periods=8).max()
        out['close_change'] = data['close'].diff()
        out['path_length'] = out['close_change'].abs().rolling(4, min_periods=4).sum()
        out['net_progress'] = data['close'].diff(4)
        out['efficiency'] = (
            out['net_progress'].div(out['path_length'].replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        out['entry'] = (
            out['prior_ceiling'].notna()
            & out['low_streak'].ge(params.streak_min)
            & data['close'].gt(out['prior_ceiling'])
            & out['efficiency'].ge(params.efficiency_min)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BreakoutParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                target[i] = 1
            elif entries[i]:
                target[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({'signal': target, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = BreakoutParams()
    index = pd.date_range('2020-01-01', periods=params.hold_bars + 4, freq='B')
    data = pd.DataFrame(index=index)
    for entries in ([True] + [False] * (len(index) - 1), [True] * len(index)):
        indicators = pd.DataFrame({'entry': entries}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].iloc[0] == 0
        assert result['signal'].iloc[1:params.hold_bars + 1].eq(1).all()
        assert result['signal'].iloc[params.hold_bars + 1] == 0
        assert result['signal'].iloc[params.hold_bars + 2] == int(entries[-1])
        assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) == 9
