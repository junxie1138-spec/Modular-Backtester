from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapContractionParams:
    gap_window: int = 12
    body_window: int = 15
    atr_window: int = 14
    gap_threshold: float = 1.2
    noise_ratio: float = 0.9
    body_threshold: float = 0.25
    atr_mult: float = 3.0
    max_hold: int = 18

    def __post_init__(self):
        if min(self.gap_window, self.body_window, self.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if self.max_hold < 1:
            raise ValueError('max_hold must be positive.')
        if not all(np.isfinite(x) and x > 0 for x in (
            self.gap_threshold, self.noise_ratio, self.atr_mult
        )):
            raise ValueError('Threshold, noise ratio, and ATR multiplier must be finite and positive.')
        if not np.isfinite(self.body_threshold) or not 0 < self.body_threshold <= 1:
            raise ValueError('body_threshold must lie in (0, 1].')


class GeneratedStrategy(BaseStrategy[GapContractionParams]):
    strategy_id = 'gen_local-w2_1791345137'

    @classmethod
    def params_type(cls):
        return GapContractionParams

    def warmup_bars(self, params):
        return int(max(2 * params.gap_window + 1, params.body_window, params.atr_window + 1))

    def indicators(self, data, params):
        prices = data[['open', 'high', 'low', 'close']].astype(float).replace([np.inf, -np.inf], np.nan)
        prices = prices.where(prices > 0)
        previous = prices['close'].shift(1)
        out = pd.DataFrame(index=data.index)
        out['gap'] = prices['open'].div(previous).sub(1)
        out['gap_mean'] = out['gap'].rolling(params.gap_window).mean()
        out['gap_noise'] = out['gap'].rolling(params.gap_window).std(ddof=0)
        out['prior_gap_noise'] = out['gap_noise'].shift(params.gap_window)
        out['gap_strength'] = out['gap_mean'].div(out['gap_noise'].replace(0, np.nan)) * np.sqrt(params.gap_window)
        out['body'] = prices['close'].div(prices['open']).sub(1)
        out['body_drift'] = out['body'].rolling(params.body_window).sum()
        out['body_activity'] = out['body'].abs().rolling(params.body_window).sum()
        out['body_efficiency'] = out['body_drift'].div(out['body_activity'].replace(0, np.nan))
        out['true_range'] = pd.concat([
            prices['high'] - prices['low'],
            (prices['high'] - previous).abs(),
            (prices['low'] - previous).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        contracting = out['gap_noise'].le(params.noise_ratio * out['prior_gap_noise'])
        long_entry = contracting & out['gap_strength'].ge(params.gap_threshold) & out['body_efficiency'].ge(params.body_threshold)
        short_entry = contracting & out['gap_strength'].le(-params.gap_threshold) & out['body_efficiency'].le(-params.body_threshold)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                stop = np.nan
                continue
            if position:
                age += 1
                # The position begins on the bar after its entry decision.
                high_water = close if age == 1 else max(high_water, close)
                low_water = close if age == 1 else min(low_water, close)
                if position == 1:
                    candidate = high_water - params.atr_mult * atr
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close <= stop
                else:
                    # Mirror the high-water trail for shorts with a low-water trail.
                    candidate = low_water + params.atr_mult * atr
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.max_hold:
                    position = 0
                    age = 0
                    stop = np.nan
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                stop = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapContractionParams(atr_mult=2.0)
    index = pd.date_range('2020-01-01', periods=6)
    for direction in (1, -1):
        closes = np.array([100., 101., 102., 100., 99., 98.])
        if direction == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({'close': closes}, index=index)
        indicators = pd.DataFrame({'atr': [1., 1., 10., 10., 1., 1.], 'entry': [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
        limited = GapContractionParams(atr_mult=2.0, max_hold=3)
        result = strategy.generate_signals(data, indicators, None, limited).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0, 0]
    assert strategy.warmup_bars(params) >= 2 * params.gap_window + 1
    flat = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1000.}, index=pd.date_range('2020-01-01', periods=60))
    derived = strategy.indicators(flat, params)
    assert derived['entry'].eq(0).all()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
