from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalFractionalParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    seasonal_window: int = 252
    entry_z: float = 1.0
    hold_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if self.fractional_window < 2 or self.seasonal_window < 30:
            raise ValueError('Insufficient indicator windows')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0.0:
            raise ValueError('entry_z must be positive and finite')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five')


class GeneratedStrategy(BaseStrategy[SeasonalFractionalParams]):
    strategy_id = 'gen_local-w2_1791308934'

    @classmethod
    def params_type(cls):
        return SeasonalFractionalParams

    def warmup_bars(self, params):
        return int(max(200, params.fractional_window + params.seasonal_window))

    def indicators(self, data, params):
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        fractional = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase the window to retain longer tails.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        window = params.seasonal_window
        complete = fractional.notna().rolling(window, min_periods=window).sum().shift(1).eq(window)
        phase_mean = pd.Series(np.nan, index=data.index, dtype=float)
        phase_std = phase_mean.copy()
        phase_count = phase_mean.copy()
        for day in range(7):
            phase = weekday.eq(day)
            observed = phase & fractional.notna()
            count = observed.astype(float).rolling(window, min_periods=window).sum().shift(1)
            values = fractional.where(observed, 0.0)
            total = values.rolling(window, min_periods=window).sum().shift(1)
            squared = values.pow(2).rolling(window, min_periods=window).sum().shift(1)
            denominator = count.where(count >= 4.0)
            mean = total / denominator
            variance = ((squared - total.pow(2) / denominator) / (denominator - 1.0)).clip(lower=0.0)
            phase_mean.loc[phase] = mean.loc[phase]
            phase_std.loc[phase] = np.sqrt(variance.loc[phase])
            phase_count.loc[phase] = count.loc[phase]
        out['seasonal_count'] = phase_count.where(complete)
        out['seasonal_mean'] = phase_mean.where(complete)
        out['seasonal_std'] = phase_std.where(complete)
        out['seasonal_z'] = ((fractional - out['seasonal_mean']) / out['seasonal_std'].where(out['seasonal_std'] > 0.0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        z = indicators['seasonal_z']
        ma = indicators['ma200']
        close = data['close']
        ready = z.notna() & ma.notna() & close.notna()
        long_entry = ready & close.gt(ma) & z.le(-params.entry_z)
        short_entry = ready & close.lt(ma) & z.ge(params.entry_z)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                signals[bar] = position
                continue
            if entries[bar] != 0:
                position = int(entries[bar])
                exit_bar = bar + params.hold_bars
            signals[bar] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = SeasonalFractionalParams()
    index = pd.date_range('2020-01-01', periods=7, freq='B')
    for direction in (1, -1):
        data = pd.DataFrame({'close': 100.0 + direction}, index=index)
        indicators = pd.DataFrame({'ma200': 100.0, 'seasonal_z': -2.0 * direction}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data['signal'].tolist() == [0, direction, direction, direction, direction, 0, direction]
        assert result.data['size'].gt(0.0).all()
