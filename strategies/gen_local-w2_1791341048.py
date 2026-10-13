from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecoveryParams:
    excursion_window: int = 10
    complexity_window: int = 6
    recovery_window: int = 3
    minimum_excursion: float = 0.006
    maximum_entropy: float = 0.85
    minimum_efficiency: float = 0.25


class GeneratedStrategy(BaseStrategy[OrdinalRecoveryParams]):
    strategy_id = 'gen_local-w2_1791341048'

    @classmethod
    def params_type(cls):
        return OrdinalRecoveryParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.excursion_window, params.complexity_window + 3, params.recovery_window + 1))

    @staticmethod
    def indicators(data, params):
        if min(params.excursion_window, params.complexity_window, params.recovery_window) < 3:
            raise ValueError('Lookback windows must be at least three.')
        if not (0 < params.minimum_excursion < 1 and 0 < params.maximum_entropy <= 1 and 0 < params.minimum_efficiency <= 1):
            raise ValueError('Invalid signal thresholds.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag2'] = out['return'].shift(2)
        out['return_lag1'] = out['return'].shift(1)
        a, b, c = out['return_lag2'], out['return_lag1'], out['return']
        valid = a.notna() & b.notna() & c.notna()
        # Equal values are ordered chronologically, giving deterministic ties.
        out['pattern'] = np.select(
            [(a <= b) & (b <= c), (a <= c) & (c < b),
             (b < a) & (a <= c), (b <= c) & (c < a),
             (c < a) & (a <= b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan
        )
        out['pattern'] = out['pattern'].where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for pattern in range(6):
            name = f'pattern_probability_{pattern}'
            out[name] = out['pattern'].eq(pattern).astype(float).where(valid).rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            probability = out[name]
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out['entropy'] = entropy / np.log(6.0)
        out['peak'] = close.rolling(params.excursion_window, min_periods=params.excursion_window).max()
        out['trough'] = close.rolling(params.excursion_window, min_periods=params.excursion_window).min()
        out['drawdown'] = 1.0 - close / out['peak']
        out['rally'] = close / out['trough'] - 1.0
        out['net_return'] = out['return'].rolling(params.recovery_window, min_periods=params.recovery_window).sum()
        out['return_path'] = out['return'].abs().rolling(params.recovery_window, min_periods=params.recovery_window).sum()
        out['efficiency'] = (out['net_return'] / out['return_path'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        organized = out['entropy'].le(params.maximum_entropy)
        long_entry = organized & out['drawdown'].ge(params.minimum_excursion) & out['efficiency'].ge(params.minimum_efficiency)
        short_entry = organized & out['rally'].ge(params.minimum_excursion) & out['efficiency'].le(-params.minimum_efficiency)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out['strength'] = ((1.0 - out['entropy']) * out['efficiency'].abs()).clip(0.0, 1.0).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.1, dtype=float)
        position = 0
        position_size = 0.1
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        strengths = indicators['strength'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        for i, entry in enumerate(entries):
            if position != 0:
                if entry == -position:
                    # Opposite entry exits first; a later bar may enter again.
                    position = 0
                    position_size = 0.1
            elif entry != 0:
                position = int(entry)
                position_size = float(0.1 + 0.9 * strengths[i])
            signals[i] = position
            sizes[i] = position_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.1).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OrdinalRecoveryParams()
    index = pd.date_range('2020-01-01', periods=40)
    close = pd.Series(100.0 + np.sin(np.arange(40)) + np.arange(40) * 0.05, index=index)
    data = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0})
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry'].iloc[:params.complexity_window + 2].eq(0).all()
    sample = data.iloc[:5]
    decisions = pd.DataFrame({'entry': [1, 0, -1, -1, 1], 'strength': [0.6, 0.0, 0.8, 0.8, 0.6]}, index=sample.index)
    result = GeneratedStrategy.generate_signals(sample, decisions, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1]
    assert np.allclose(result['size'], [0.1, 0.64, 0.64, 0.1, 0.82])
