from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapResponseParams:
    regime_window: int = 40
    atr_window: int = 14
    min_correlation: float = 0.20
    entry_score: float = 0.40
    trail_atr: float = 0.80
    max_hold: int = 2

    def __post_init__(self):
        if self.regime_window < 3 or self.atr_window < 2:
            raise ValueError('Indicator windows are too short')
        if not 0.0 < self.min_correlation < 1.0:
            raise ValueError('min_correlation must be between zero and one')
        if not np.isfinite(self.entry_score) or self.entry_score <= 0.0:
            raise ValueError('entry_score must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0.0:
            raise ValueError('trail_atr must be finite and positive')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions')


class GeneratedStrategy(BaseStrategy[GapResponseParams]):
    strategy_id = 'gen_local_1791208746'

    @classmethod
    def params_type(cls):
        return GapResponseParams

    def warmup_bars(self, params):
        return int(max(params.regime_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['gap'] = data['open'].div(previous_close.where(previous_close > 0)).sub(1)
        out['session_return'] = data['close'].div(data['open'].where(data['open'] > 0)).sub(1)
        out['lagged_gap'] = out['gap'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        window = params.regime_window
        out['covariance'] = out['lagged_gap'].rolling(window).cov(out['session_return'])
        out['gap_variance'] = out['lagged_gap'].rolling(window).var()
        out['session_variance'] = out['session_return'].rolling(window).var()
        out['correlation_scale'] = np.sqrt(out['gap_variance'] * out['session_variance'])
        out['correlation'] = out['covariance'].div(out['correlation_scale'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out['gap_mean'] = out['gap'].rolling(window).mean().shift(1)
        out['gap_scale'] = out['gap'].rolling(window).std().shift(1)
        out['gap_z'] = (out['gap'] - out['gap_mean']).div(out['gap_scale'].replace(0.0, np.nan))
        out['score'] = out['correlation'] * out['gap_z']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({'signal': 0, 'size': 0.20}, index=data.index)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        scores = indicators['score'].to_numpy(dtype=float)
        correlations = indicators['correlation'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        exposure = 0.20
        for i in range(len(data)):
            close, atr, score, correlation = closes[i], atrs[i], scores[i], correlations[i]
            valid = np.isfinite([close, atr, score, correlation]).all() and close > 0.0 and atr > 0.0
            if active:
                if not valid:
                    active = False
                    continue
                # The first held close occurs one bar after the entry decision.
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= params.max_hold:
                    active = False
                    continue
            elif valid and abs(correlation) >= params.min_correlation and score >= params.entry_score:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                exposure = float(0.20 + 0.75 * np.clip(score / 1.50, 0.0, 1.0))
            if active:
                signals[i] = 1
                sizes[i] = exposure
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [100., 101., 99., 100., 100., 100., 100.]}, index=index)
    derived = pd.DataFrame({
        'atr': [1.] * 7,
        'correlation': [0.5] * 7,
        'score': [1.5, 0., 0., 0.5, 0., 0., 0.],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, GapResponseParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert result['size'].iloc[1] > result['size'].iloc[4]
    derived.loc[index[0], 'correlation'] = -0.5
    derived.loc[index[0], 'score'] = 1.5
    assert strategy.generate_signals(data, derived, None, GapResponseParams()).data['signal'].iloc[1] == 1
