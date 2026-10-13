from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    lookback: int = 126
    streak_percentile: float = 0.90
    profit_target: float = 0.02
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.lookback < 3:
            raise ValueError('lookback must be at least 3')
        if not 0.0 < self.streak_percentile < 1.0:
            raise ValueError('streak_percentile must be between zero and one')
        if self.profit_target <= 0.0 or not np.isfinite(self.profit_target):
            raise ValueError('profit_target must be finite and positive')
        if self.max_hold_bars < 1:
            raise ValueError('max_hold_bars must be positive')


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = 'gen_local_1791260711'

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & data['close'].gt(0))
        out['return'] = close.div(close.shift(1)).sub(1)
        direction = out['return'].gt(0).astype(int) - out['return'].lt(0).astype(int)
        out['direction'] = direction
        groups = direction.ne(direction.shift(1)).cumsum()
        out['streak'] = direction.groupby(groups).cumcount().add(1).where(direction.ne(0), 0).clip(upper=params.lookback)
        out['streak_threshold'] = out['streak'].rolling(params.lookback, min_periods=params.lookback).quantile(params.streak_percentile).shift(1)
        out['terminal_contraction'] = out['return'].shift(1).abs().lt(out['return'].shift(2).abs())
        exhausted = out['streak'].shift(1).gt(out['streak_threshold'])
        reversed_direction = direction.mul(direction.shift(1)).eq(-1)
        ready = out['streak_threshold'].notna() & close.notna()
        entry = exhausted & reversed_direction & out['terminal_contraction'] & ready
        out['entry_signal'] = direction.where(entry, 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry_signal'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1

        for i in range(len(data)):
            if position != 0:
                if not np.isfinite(entry_price):
                    # The previous close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired:
                    # Close-observed exits execute at the following open.
                    position = 0
                    entry_price = np.nan
                signals[i] = position
            elif entries[i] != 0:
                position = int(entries[i])
                entry_price = np.nan
                signals[i] = position

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StreakParams()
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_signal': [1] + [0] * 11}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 3
    data.loc[index[2], 'close'] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1] + [0] * 9
    indicators.loc[index[0], 'entry_signal'] = -1
    data.loc[index[2], 'close'] = 97.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1] + [0] * 9
    data['close'] = [100, 99, 97, 96, 97, 97, 98, 99, 100, 99, 98, 97]
    derived = strategy.indicators(data, params)
    assert derived['streak'].iloc[:7].tolist() == [0, 1, 2, 3, 1, 0, 1]
    assert derived['entry_signal'].eq(0).all()
