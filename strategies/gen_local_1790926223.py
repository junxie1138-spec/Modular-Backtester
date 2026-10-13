from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    shape_bars: int = 5
    history_bars: int = 504
    neighbors: int = 12
    hold_bars: int = 8
    atr_bars: int = 14
    forecast_threshold: float = 0.2
    stop_atr: float = 2.0

    def __post_init__(self):
        if self.shape_bars < 2 or min(self.history_bars, self.neighbors, self.hold_bars, self.atr_bars) < 1:
            raise ValueError('Lookbacks and neighbor count must be positive; shape_bars must be at least two.')
        if self.history_bars < self.neighbors * self.hold_bars:
            raise ValueError('History must accommodate separated analogs.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        if not np.isfinite(self.forecast_threshold) or self.forecast_threshold <= 0:
            raise ValueError('forecast_threshold must be finite and positive.')


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = 'gen_local_1790926223'

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    def warmup_bars(self, params):
        return int(max(params.atr_bars + 1, params.history_bars + params.shape_bars + params.hold_bars + 1))

    def indicators(self, data, params):
        p = params
        result = pd.DataFrame(index=data.index)
        daily_range = (data['high'] - data['low']).where(lambda s: s > 0)
        midpoint = (data['high'] + data['low']) / 2.0
        mean_range = daily_range.rolling(p.shape_bars, min_periods=p.shape_bars).mean()
        range_sum = daily_range.rolling(p.shape_bars, min_periods=p.shape_bars).sum()
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        result['atr'] = true_range.rolling(p.atr_bars, min_periods=p.atr_bars).mean()
        result['range'] = daily_range
        result['midpoint'] = midpoint
        result['mean_range'] = mean_range
        result['range_sum'] = range_sum
        result['contraction'] = daily_range / mean_range
        result['deformation'] = (midpoint - midpoint.shift(p.shape_bars - 1)) / range_sum
        feature_columns = []
        for lag in range(p.shape_bars):
            name = f'range_shape_{lag}'
            result[name] = np.log(daily_range.shift(lag) / mean_range)
            feature_columns.append(name)
        result['shape_set'] = 2.0 * result['deformation']
        feature_columns.append('shape_set')
        features = result[feature_columns].to_numpy(dtype=float)
        # Labels are accessed only after their exit open has occurred.
        labels = ((data['open'].shift(-p.hold_bars - 1) - data['open'].shift(-1)) / result['atr'].where(result['atr'] > 0)).to_numpy(dtype=float)
        forecast = np.zeros(len(data), dtype=float)
        win_rate = np.zeros(len(data), dtype=float)
        match_error = np.full(len(data), np.inf, dtype=float)
        for t in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(features[t]).all():
                continue
            # Candidate holding windows do not overlap, and every label is mature.
            last_exclusive = t - p.hold_bars
            first = max(p.shape_bars - 1, last_exclusive - p.history_bars)
            candidates = np.arange(first, last_exclusive, p.hold_bars, dtype=int)
            valid = np.isfinite(features[candidates]).all(axis=1) & np.isfinite(labels[candidates])
            candidates = candidates[valid]
            if len(candidates) < p.neighbors:
                continue
            distances = np.sqrt(np.mean((features[candidates] - features[t]) ** 2, axis=1))
            nearest = np.argpartition(distances, p.neighbors - 1)[:p.neighbors]
            weights = 1.0 / (0.05 + distances[nearest])
            outcomes = labels[candidates[nearest]]
            forecast[t] = np.average(outcomes, weights=weights)
            win_rate[t] = np.average(outcomes > 0, weights=weights)
            match_error[t] = np.average(distances[nearest], weights=weights)
        result['forecast'] = forecast
        result['win_rate'] = win_rate
        result['match_error'] = match_error
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        eligible = (
            (indicators['forecast'] >= p.forecast_threshold)
            & (indicators['win_rate'] >= 0.55)
            & (indicators['match_error'] <= 0.6)
            & (indicators['contraction'] < 0.9)
            & (indicators['deformation'].abs() <= 0.35)
            & (indicators['atr'] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        strength = (
            (indicators['forecast'] / 1.25).clip(0.0, 1.0)
            * ((indicators['win_rate'] - 0.5) / 0.25).clip(0.0, 1.0)
        ).fillna(0.0)
        entry_sizes = (0.2 + 0.75 * strength).to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        pending = False
        entry_bar = 0
        stop_price = 0.0
        held_size = 1.0
        entry_atr = 0.0
        for i in range(len(data)):
            if pending:
                # Yesterday's decision fills at today's open; ATR stays frozen.
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    raise ValueError('Entry open must be finite and positive.')
                stop_price = opening[i] - p.stop_atr * entry_atr
                entry_bar = i
                active = True
                pending = False
            if active:
                sizes[i] = held_size
                if not np.isfinite(closing[i]):
                    signals[i] = 1
                    continue
                if closing[i] < stop_price or i - entry_bar + 1 >= p.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif eligible[i] and np.isfinite(closing[i]) and closing[i] > 0:
                pending = True
                entry_atr = atr[i]
                held_size = entry_sizes[i]
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 105.0, 103.0, 102.0],
        'high': [106.0] * 4,
        'low': [99.0] * 4,
        'close': [100.0, 104.0, 102.0, 101.0],
        'volume': [1000.0] * 4,
    }, index=index)
    ind = pd.DataFrame({
        'forecast': [0.4, 0.0, 0.0, 0.0],
        'win_rate': [0.75] * 4,
        'match_error': [0.1] * 4,
        'contraction': [0.8] * 4,
        'deformation': [0.0] * 4,
        'atr': [1.0] * 4,
    }, index=index)
    strategy = GeneratedStrategy()
    params = RangeAnalogParams()
    context = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, ind, context, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    ind.loc[index[0], 'forecast'] = 0.8
    stronger = strategy.generate_signals(data, ind, context, params).data
    assert 0 < result['size'].iloc[1] < stronger['size'].iloc[1] <= 0.95
