from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 32
    pressure_window: int = 40
    atr_window: int = 14
    gap_min: float = 0.5
    pressure_min: float = 0.6
    stop_atr: float = 1.6
    hold_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError('frac_order must be between zero and one')
        if min(self.frac_window, self.pressure_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two')
        if min(self.gap_min, self.pressure_min, self.stop_atr) <= 0:
            raise ValueError('Thresholds must be positive')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1790963229'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.frac_window + params.pressure_window, params.atr_window + 1))

    def indicators(self, data, params):
        close = data['close'].astype(float)
        previous = close.shift(1)
        tr = pd.concat([(data['high'] - data['low']).abs(),
                        (data['high'] - previous).abs(),
                        (data['low'] - previous).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        scale = atr.where(atr > 0)
        weights = np.ones(params.frac_window, dtype=float)
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # ponytail: finite fractional memory; increase frac_window for longer memory.
        fractional = close.copy()
        for lag in range(1, params.frac_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        center = fractional.rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        spread = fractional.rolling(params.pressure_window, min_periods=params.pressure_window).std(ddof=0)
        pressure = (fractional - center) / spread.where(spread > 0)
        gap = data['open'] - previous
        retained = (close - previous) / gap.where(gap.abs() > 0)
        result = pd.DataFrame({'atr': atr, 'fractional_close': fractional,
                               'pressure_center': center, 'pressure_spread': spread,
                               'pressure': pressure, 'gap': gap,
                               'gap_atr': gap / scale, 'retained': retained}, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        pressure = indicators['pressure']
        gap = indicators['gap_atr']
        ready = indicators[['atr', 'pressure', 'gap_atr', 'retained']].notna().all(axis=1)
        eligible = (ready & (indicators['atr'] > 0)
                    & (pressure.abs() >= params.pressure_min)
                    & (gap.abs() >= params.gap_min)
                    & (pressure * gap > 0)
                    & indicators['retained'].between(0.60, 1.15))
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[eligible & (gap > 0)] = 1
        candidate.loc[eligible & (gap < 0)] = -1
        conviction = (np.sqrt(pressure.abs() * gap.abs()) / 1.5).clip(0.2, 1.0).fillna(0.2)
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = candidate.to_numpy(dtype=int)
        strengths = conviction.to_numpy(dtype=float)
        side = 0
        pending = 0
        entry_bar = -1
        stop = np.nan
        position_size = 1.0
        for i in range(n):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    side = pending
                    entry_bar = i
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                pending = 0
            if side:
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    continue
                signals[i] = side
                sizes[i] = position_size
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = int(candidates[i])
                position_size = float(strengths[i])
                signals[i] = pending
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=8)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    for direction in (1, -1):
        for hit_stop in (False, True):
            data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0,
                                 'close': 100.0, 'volume': 1000.0}, index=index)
            features = pd.DataFrame({'atr': 1.0, 'pressure': 0.0,
                                     'gap_atr': 0.0, 'retained': 0.8}, index=index)
            features.loc[index[0], ['pressure', 'gap_atr']] = [direction * 1.2, direction * 0.6]
            if hit_stop:
                data.loc[index[3], 'close'] = 100.0 - direction * 2.0
                features.loc[index[3], 'atr'] = 10.0
            result = strategy.generate_signals(data, features, ctx, params).data
            end = 4 if hit_stop else 5
            expected = np.zeros(8, dtype=int)
            expected[1:end] = direction
            assert np.array_equal(result['signal'].to_numpy(), expected)
            assert result['size'].gt(0).all()
            assert 0.2 < result.loc[index[1], 'size'] < 1.0
    flat = strategy.generate_signals(data, features * np.nan, ctx, params).data
    assert flat['signal'].eq(0).all()
    assert strategy.warmup_bars(params) >= params.frac_window + params.pressure_window - 1


if __name__ == '__main__':
    _self_check()
