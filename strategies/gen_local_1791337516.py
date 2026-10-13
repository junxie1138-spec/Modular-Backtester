from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepertoireParams:
    pattern_window: int = 10
    range_window: int = 20
    atr_window: int = 14
    atr_multiple: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        if min(self.pattern_window, self.range_window, self.atr_window) < 3:
            raise ValueError('Indicator windows must be at least three.')
        if self.max_hold < 1 or not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('Holding period and ATR multiple must be positive.')


class GeneratedStrategy(BaseStrategy[RepertoireParams]):
    strategy_id = 'gen_local_1791337516'

    @classmethod
    def params_type(cls):
        return RepertoireParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.pattern_window + 7, params.range_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['previous_close'] = close.shift(1)
        out['older_close'] = close.shift(2)
        a, b, c = out['older_close'], out['previous_close'], close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out['pattern_code'] = (4 * a.lt(b).astype(int) + 2 * a.lt(c).astype(int) + b.lt(c).astype(int)).where(valid)
        count_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = f'pattern_count_{code}'
            out[column] = out['pattern_code'].eq(code).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).sum()
            count_columns.append(column)
        counts = out[count_columns]
        out['complexity'] = counts.gt(0).sum(axis=1).astype(float).where(counts.notna().all(axis=1))
        out['previous_complexity'] = out['complexity'].shift(5)
        out['range_low'] = close.rolling(params.range_window, min_periods=params.range_window).min()
        out['range_high'] = close.rolling(params.range_window, min_periods=params.range_window).max()
        out['range_span'] = (out['range_high'] - out['range_low']).replace(0, np.nan)
        out['position'] = (close - out['range_low']) / out['range_span']
        out['previous_position'] = out['position'].shift(1)
        out['high_low_range'] = data['high'] - data['low']
        out['high_gap_range'] = (data['high'] - out['previous_close']).abs()
        out['low_gap_range'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['high_low_range', 'high_gap_range', 'low_gap_range']].max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ordinal_agreement'] = out['complexity'].ge(5) & out['previous_complexity'].le(4)
        out['position_agreement'] = out['previous_position'].lt(0.5) & out['position'].ge(0.5) & out['position'].le(0.8)
        out['entry'] = out['ordinal_agreement'] & out['position_agreement'] & out['atr'].gt(0) & np.isfinite(out['atr'])
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            if active:
                age += 1
                if np.isfinite(close[i]):
                    high_water = max(high_water, close[i])
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, high_water - params.atr_multiple * atr[i])
                    stopped = close[i] <= stop
                else:
                    stopped = False
                if stopped or age >= params.max_hold:
                    active = False
                else:
                    signal[i] = 1
            elif entry[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                age = 0
                # Entry fills next bar; start the close high-water mark there.
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100., 101., 105., 103., 101., 100., 104., 103.]}, index=index)
    derived = pd.DataFrame({'atr': [2., 2., 1., 4., 4., 1., 1., 1.], 'entry': [True, False, False, False, False, False, True, False]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, RepertoireParams(atr_multiple=2.0)).data
    # ATR expansion must not lower the stop; exits also shift exactly once.
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0, 1]
    result = strategy.generate_signals(data, derived, None, RepertoireParams(atr_multiple=100.0, max_hold=2)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 1]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
