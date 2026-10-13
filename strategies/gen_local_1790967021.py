from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionAccelerationParams:
    roc_bars: int = 15
    smooth_bars: int = 5
    compression_bars: int = 10
    baseline_bars: int = 60
    compression_ratio: float = 0.85
    acceleration_threshold: float = 0.75


class GeneratedStrategy(BaseStrategy[CompressionAccelerationParams]):
    strategy_id = 'gen_local_1790967021'

    @classmethod
    def params_type(cls) -> type[CompressionAccelerationParams]:
        return CompressionAccelerationParams

    def warmup_bars(self, params: CompressionAccelerationParams) -> int:
        return int(max(2 * params.roc_bars + params.smooth_bars, params.baseline_bars + 1, params.compression_bars))

    def indicators(self, data: pd.DataFrame, params: CompressionAccelerationParams) -> pd.DataFrame:
        if min(params.roc_bars, params.smooth_bars, params.compression_bars, params.baseline_bars) < 1:
            raise ValueError('Lookbacks must be positive integers')
        if not (0 < params.compression_ratio <= 1) or not (np.isfinite(params.acceleration_threshold) and params.acceleration_threshold > 0):
            raise ValueError('Invalid compression ratio or acceleration threshold')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['daily_return'] = close.pct_change(fill_method=None)
        out['daily_range'] = (data['high'] - data['low']).where(data['high'] >= data['low']) / close
        out['short_range'] = out['daily_range'].rolling(params.compression_bars, min_periods=params.compression_bars).mean()
        out['baseline_range'] = out['daily_range'].rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out['compression'] = out['short_range'] / out['baseline_range'].where(out['baseline_range'] > 0)
        out['roc'] = close.pct_change(periods=params.roc_bars, fill_method=None)
        out['raw_acceleration'] = out['roc'].diff(params.roc_bars)
        out['smoothed_acceleration'] = out['raw_acceleration'].rolling(params.smooth_bars, min_periods=params.smooth_bars).mean()
        out['return_scale'] = out['daily_return'].abs().rolling(params.baseline_bars, min_periods=params.baseline_bars).mean() * np.sqrt(params.roc_bars)
        out['acceleration'] = out['smoothed_acceleration'] / out['return_scale'].where(out['return_scale'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionAccelerationParams) -> SignalFrame:
        acceleration = indicators['acceleration']
        compression = indicators['compression']
        eligible = np.isfinite(acceleration) & np.isfinite(compression) & (compression <= params.compression_ratio)
        entry = np.where(eligible, np.where(acceleration >= params.acceleration_threshold, 1, np.where(acceleration <= -params.acceleration_threshold, -1, 0)), 0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entry):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = CompressionAccelerationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'compression': 0.5, 'acceleration': [np.nan, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    indicators['acceleration'] *= -1
    mirrored = strategy.generate_signals(data, indicators, None, params).data
    assert mirrored['signal'].equals(-result['signal'])
    indicators['compression'] = 1.5
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()
