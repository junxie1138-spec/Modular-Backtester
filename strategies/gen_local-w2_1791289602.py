from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    efficiency_window: int = 20
    streak_length: int = 3
    pressure_window: int = 2
    trend_threshold: float = 0.40
    range_threshold: float = 0.20
    pressure_threshold: float = 0.15
    profit_target: float = 0.006
    max_hold_bars: int = 2

    def __post_init__(self):
        if min(self.efficiency_window, self.streak_length, self.pressure_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 0 <= self.range_threshold < self.trend_threshold <= 1:
            raise ValueError('Invalid efficiency thresholds.')
        if not 0 < self.pressure_threshold <= 1:
            raise ValueError('Invalid pressure threshold.')
        if not 0 < self.profit_target < 1 or self.max_hold_bars not in (1, 2):
            raise ValueError('Invalid exit parameters.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791289602'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.efficiency_window + 1, params.streak_length + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['change'] = close.diff()
        out['direction'] = np.sign(out['change']).fillna(0).astype(int)
        # Capped streaks depend only on the most recent streak_length changes.
        direction = out['direction']
        groups = direction.ne(direction.shift()).cumsum()
        out['streak'] = (direction.groupby(groups).cumcount() + 1).clip(upper=params.streak_length).where(direction.ne(0), 0)
        out['path_length'] = out['change'].abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        out['displacement'] = close.diff(params.efficiency_window).abs()
        out['efficiency'] = out['displacement'].div(out['path_length'].replace(0, np.nan)).clip(0, 1)
        out['bar_range'] = data['high'] - data['low']
        out['close_location'] = ((2 * close - data['high'] - data['low']) / out['bar_range'].where(out['bar_range'] > 0)).clip(-1, 1)
        out['pressure'] = out['close_location'].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out['regime'] = np.select([out['efficiency'].ge(params.trend_threshold), out['efficiency'].le(params.range_threshold)], [1, -1], default=0).astype(int)
        out['streak_vote'] = (direction * out['regime']).where(out['streak'].ge(params.streak_length), 0).astype(int)
        out['pressure_vote'] = np.select([out['pressure'].ge(params.pressure_threshold), out['pressure'].le(-params.pressure_threshold)], [1, -1], default=0).astype(int)
        valid = out[['efficiency', 'pressure']].notna().all(axis=1)
        valid &= data[['open', 'high', 'low', 'close']].gt(0).all(axis=1)
        valid &= np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        out['entry'] = out['streak_vote'].where(valid & out['streak_vote'].eq(out['pressure_vote']), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # Yesterday's decision fills at today's open; never read tomorrow's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                if not valid_entry or target_hit or age >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    age = 0
                    # Exit at next open; do not reenter on the exit decision bar.
                    decisions[i] = 0
                    continue
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                age = 0
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, -1, 0, 0]
    data.loc[index[4], 'close'] = 99.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0, 0]
    assert result['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived['entry'].eq(0).all()
    assert derived.index.equals(data.index)
    assert strategy.warmup_bars(params) >= params.efficiency_window + 1
