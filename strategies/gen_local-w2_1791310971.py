from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiscoveryParams:
    roc_bars: int = 3
    regime_window: int = 12
    percentile_window: int = 126
    atr_window: int = 14
    acceleration_quantile: float = 0.85
    stop_atr: float = 1.25
    hold_bars: int = 2

    def __post_init__(self):
        if min(self.roc_bars, self.regime_window, self.percentile_window, self.atr_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 0.5 < self.acceleration_quantile < 1.0:
            raise ValueError('Acceleration quantile must be between 0.5 and 1.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('Stop multiplier must be positive and finite.')
        if self.hold_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[DiscoveryParams]):
    strategy_id = 'gen_local-w2_1791310971'

    @classmethod
    def params_type(cls):
        return DiscoveryParams

    def warmup_bars(self, params):
        return int(max(
            params.roc_bars + params.percentile_window + 2,
            params.regime_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['roc'] = data['close'].pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['gap_return'] = data['open'] / out['previous_close'].replace(0, np.nan) - 1.0
        out['session_return'] = data['close'] / data['open'].replace(0, np.nan) - 1.0
        out['gap_energy'] = out['gap_return'].pow(2).rolling(params.regime_window).sum()
        out['session_energy'] = out['session_return'].pow(2).rolling(params.regime_window).sum()
        total_energy = out['gap_energy'] + out['session_energy']
        out['overnight_share'] = out['gap_energy'] / total_energy.replace(0, np.nan)
        out['regime_threshold'] = out['overnight_share'].rolling(params.percentile_window).quantile(0.5).shift(1)
        out['acceleration_threshold'] = out['acceleration'].abs().rolling(params.percentile_window).quantile(params.acceleration_quantile).shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        required = ['acceleration', 'acceleration_threshold', 'overnight_share', 'regime_threshold', 'atr']
        valid = np.isfinite(indicators[required]).all(axis=1)
        valid &= np.isfinite(data['close']) & (data['close'] > 0)
        valid &= (indicators['atr'] > 0) & (indicators['acceleration_threshold'] >= 0)
        extreme = indicators['acceleration'].abs() > indicators['acceleration_threshold']
        continuation = indicators['overnight_share'] > indicators['regime_threshold']
        direction = np.sign(indicators['acceleration'].fillna(0))
        candidates = np.where(valid & extreme, direction * np.where(continuation, 1, -1), 0).astype(int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        stop = np.nan

        for i in range(len(data)):
            # A prior close's instruction fills at this session's open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(decisions[i - 1])
                    age = 0
                    # Both the entry price and ATR are frozen for this trade.
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]

            if position != 0:
                age += 1
                stopped = not np.isfinite(closes[i]) or closes[i] <= 0
                if not stopped:
                    stopped = closes[i] < stop if position == 1 else closes[i] > stop
                if stopped or age >= params.hold_bars:
                    position = 0
                    decisions[i] = 0
                    continue
                decisions[i] = position
            else:
                decisions[i] = candidates[i]

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    p = DiscoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 104.0, 'low': 96.0,
        'close': [100.0, 97.0, 100.0, 103.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'acceleration': [2.0, 0.0, -2.0, 0.0, 2.0, 0.0],
        'acceleration_threshold': 1.0,
        'overnight_share': 0.8,
        'regime_threshold': 0.5,
        'atr': 1.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].tolist() == [0, 1, 0, -1, 0, 1]
    data['close'] = 100.0
    indicators['acceleration'] = 2.0
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    indicators['overnight_share'] = 0.2
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result['signal'].tolist() == [0, -1, -1, 0, -1, -1]
    indicators['acceleration_threshold'] = np.nan
    assert strategy.generate_signals(data, indicators, None, p).data['signal'].eq(0).all()
