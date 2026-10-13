from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPatternParams:
    pattern_window: int = 18
    hold_bars: int = 4

    def __post_init__(self):
        if isinstance(self.pattern_window, bool) or not isinstance(self.pattern_window, int) or self.pattern_window < 6:
            raise ValueError('pattern_window must be an integer >= 6')
        if isinstance(self.hold_bars, bool) or not isinstance(self.hold_bars, int) or not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be an integer from 3 to 5')


class GeneratedStrategy(BaseStrategy[GapPatternParams]):
    strategy_id = 'gen_local-w2_1790971471'

    @classmethod
    def params_type(cls):
        return GapPatternParams

    def warmup_bars(self, params):
        return int(params.pattern_window + 4)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & (data['close'] > 0))
        opening = data['open'].where(np.isfinite(data['open']) & (data['open'] > 0))
        out['previous_close'] = close.shift(1)
        out['gap'] = (opening / out['previous_close'] - 1.0).replace([np.inf, -np.inf], np.nan)
        out['gap_lag1'] = out['gap'].shift(1)
        out['gap_lag2'] = out['gap'].shift(2)
        a, b, c = out['gap_lag2'], out['gap_lag1'], out['gap']
        valid = a.notna() & b.notna() & c.notna()
        # Pairwise comparisons encode the six permutations; ties favor earlier observations.
        out['pattern'] = (4 * a.le(b).astype(int) + 2 * a.le(c).astype(int) + b.le(c).astype(int)).where(valid)
        codes = (0, 1, 3, 4, 6, 7)
        probability_columns = []
        for code in codes:
            name = f'probability_{code}'
            occurrence = out['pattern'].eq(code).astype(float).where(valid)
            out[name] = occurrence.shift(1).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            probability_columns.append(name)
        probabilities = out[probability_columns]
        # clip handles 0 * log(0); incomplete historical windows remain NaN.
        out['entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out['pattern_probability'] = np.nan
        for code, name in zip(codes, probability_columns):
            matching = out['pattern'].eq(code)
            out.loc[matching, 'pattern_probability'] = out.loc[matching, name]
        out['retracement'] = ((opening - close) / (opening - out['previous_close']).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        eligible = (
            valid
            & out['entropy'].lt(0.85)
            & out['pattern_probability'].le(1.0 / params.pattern_window)
            & out['retracement'].gt(0.0)
            & out['retracement'].lt(1.0)
        )
        out['entry'] = (-np.sign(out['gap'].fillna(0.0))).where(eligible, 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    # Preserve an explicit flat bar, even if another entry qualifies.
                    continue
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GapPatternParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert result['size'].gt(0).all()
    cold = GeneratedStrategy.indicators(None, data, params)
    assert cold['entry'].eq(0).all()
