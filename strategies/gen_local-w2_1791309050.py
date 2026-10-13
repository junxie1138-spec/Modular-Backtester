from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TensionParams:
    tension_threshold: float = 1.5
    profit_target: float = 0.025

    def __post_init__(self):
        if not np.isfinite(self.tension_threshold) or self.tension_threshold <= 0:
            raise ValueError('tension_threshold must be finite and positive')
        if not np.isfinite(self.profit_target) or not 0 < self.profit_target < 1:
            raise ValueError('profit_target must be between zero and one')


class GeneratedStrategy(BaseStrategy[TensionParams]):
    strategy_id = 'gen_local-w2_1791309050'

    @classmethod
    def params_type(cls):
        return TensionParams

    def warmup_bars(self, params: TensionParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: TensionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(10, min_periods=10).mean()
        out['channel_high'] = data['high'].rolling(20, min_periods=20).max()
        out['channel_low'] = data['low'].rolling(20, min_periods=20).min()
        out['midpoint'] = (out['channel_high'] + out['channel_low']) / 2.0
        out['displacement'] = data['close'] - out['midpoint']
        out['tension'] = out['displacement'] / out['atr'].where(out['atr'] > 0)
        out['previous_atr'] = out['atr'].shift(1)
        out['previous_displacement'] = out['displacement'].shift(1)
        out['previous_tension'] = out['tension'].shift(1)
        ready = np.isfinite(out[['atr', 'displacement', 'tension',
                                 'previous_atr', 'previous_displacement',
                                 'previous_tension']]).all(axis=1)
        trigger = (
            ready
            & (out['tension'].abs() >= params.tension_threshold)
            & (out['previous_tension'].abs() < params.tension_threshold)
            & (out['atr'] < out['previous_atr'])
            & (out['displacement'].abs() < out['previous_displacement'].abs())
            & (np.sign(out['displacement']) == np.sign(out['previous_displacement']))
        )
        out['entry'] = np.where(trigger, -np.sign(out['displacement']), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TensionParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0
        for i in range(len(data)):
            if position:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= 8:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                held_bars = 0
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = TensionParams()
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [-1] + [0] * 9}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [-1] * 8 + [0]
    indicators['entry'] = [1] + [0] * 9
    data.loc[index[1], 'close'] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1] + [0] * 8
