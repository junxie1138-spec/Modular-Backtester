from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_bars: int = 3
    pressure_bars: int = 3
    pressure_threshold: float = 0.15
    holding_bars: int = 4

    def __post_init__(self):
        if not isinstance(self.streak_bars, int) or self.streak_bars < 1:
            raise ValueError('streak_bars must be a positive integer')
        if not isinstance(self.pressure_bars, int) or self.pressure_bars < 1:
            raise ValueError('pressure_bars must be a positive integer')
        if not 0 < self.pressure_threshold <= 1:
            raise ValueError('pressure_threshold must be in (0, 1]')
        if not isinstance(self.holding_bars, int) or not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be an integer from 3 to 5')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791341652'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.streak_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        delta = data['close'].diff()
        up = delta.gt(0)
        down = delta.lt(0)
        # Counts saturate at the entry threshold, bounding required history.
        up_streak = up.groupby((~up).cumsum()).cumsum().clip(upper=params.streak_bars)
        down_streak = down.groupby((~down).cumsum()).cumsum().clip(upper=params.streak_bars)
        span = data['high'] - data['low']
        location = ((2 * data['close'] - data['high'] - data['low']) / span.where(span > 0)).clip(-1, 1)
        location = location.mask(span.eq(0), 0.0)
        volume = data['volume'].clip(lower=0)
        weighted = location * volume
        total = volume.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        pressure = weighted.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum() / total.where(total > 0)
        return pd.DataFrame({
            'close_change': delta,
            'up_streak': up_streak,
            'down_streak': down_streak,
            'daily_range': span,
            'close_location': location,
            'nonnegative_volume': volume,
            'weighted_pressure': weighted,
            'window_volume': total,
            'pressure': pressure,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        long_entry = indicators['down_streak'].ge(params.streak_bars) & indicators['pressure'].ge(params.pressure_threshold)
        short_entry = indicators['up_streak'].ge(params.streak_bars) & indicators['pressure'].le(-params.pressure_threshold)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        target = np.zeros(len(data), dtype=int)
        position, exit_bar = 0, -1
        for i, entry in enumerate(candidate):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    target[i] = position
                # The expiry bar stays flat even if another entry qualifies.
                continue
            if entry:
                position = int(entry)
                exit_bar = i + params.holding_bars
                target[i] = position
        df = pd.DataFrame({'signal': target, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 10.0, 'high': 11.0, 'low': 9.0, 'close': 10.0, 'volume': 100.0}, index=index)
    forced = pd.DataFrame({
        'down_streak': [3, 3] + [0] * 10,
        'up_streak': [0, 0] + [3] * 10,
        'pressure': [0.5, 0.5] + [-0.5] * 10,
    }, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    for holding in (3, 4, 5):
        p = StrategyParams(holding_bars=holding)
        signal = strategy.generate_signals(data, forced, None, p).data['signal']
        assert signal.iloc[1:1 + holding].eq(1).all()
        assert signal.iloc[1 + holding] == 0
    disagrees = forced.copy()
    disagrees['pressure'] = 0.0
    assert strategy.generate_signals(data, disagrees, None, params).data['signal'].eq(0).all()
    data['high'] = data['low'] = data['close']
    data['volume'] = 0.0
    computed = strategy.indicators(data, params)
    assert strategy.generate_signals(data, computed, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
