from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_streak: int = 3
    repair_window: int = 5
    peak_window: int = 20
    min_drawdown: float = 0.01
    hold_bars: int = 2

    def __post_init__(self):
        if self.min_streak < 1 or self.repair_window < 2 or self.peak_window < 2:
            raise ValueError('Lookbacks must be positive; repair and peak windows must be at least two.')
        if not 0.0 < self.min_drawdown < 1.0:
            raise ValueError('min_drawdown must be between zero and one.')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791316551'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.repair_window + 2, params.min_streak + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        volume = data['volume'].where(data['volume'] >= 0)
        out['change'] = close.diff()
        out['declining'] = out['change'].lt(0).fillna(False)
        groups = (~out['declining']).cumsum()
        out['decline_streak'] = out['declining'].astype(int).groupby(groups).cumsum().clip(upper=params.min_streak)
        out['prior_streak'] = out['decline_streak'].shift(1)
        out['streak_ended'] = (out['prior_streak'].ge(params.min_streak) & out['change'].gt(0)).fillna(False)
        out['weighted_close'] = close * volume
        out['volume_sum'] = volume.rolling(params.repair_window).sum()
        out['weighted_sum'] = out['weighted_close'].rolling(params.repair_window).sum()
        out['repair_reference'] = (out['weighted_sum'] / out['volume_sum'].replace(0, np.nan)).shift(1)
        out['previous_reference'] = out['repair_reference'].shift(1)
        out['previous_close'] = close.shift(1)
        out['reclaimed'] = (close.gt(out['repair_reference']) & out['previous_close'].le(out['previous_reference'])).fillna(False)
        out['peak'] = close.rolling(params.peak_window).max()
        out['drawdown'] = 1.0 - close / out['peak'].replace(0, np.nan)
        out['ready'] = np.arange(len(data)) >= self.warmup_bars(params) - 1
        out['entry'] = (out['streak_ended'] & out['reclaimed'] & out['drawdown'].ge(params.min_drawdown) & out['ready']).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': [True, True, True, True, False, True, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    for hold, expected in ((1, [0, 1, 0, 1, 0, 0, 1, 0, 0]), (2, [0, 1, 1, 0, 1, 1, 0, 0, 0])):
        result = strategy.generate_signals(data, indicators, None, RecoveryParams(hold_bars=hold)).data
        assert result['signal'].tolist() == expected
        assert result['size'].gt(0).all()
