from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    channel_bars: int = 2
    area_threshold: float = 0.15

    def __post_init__(self):
        if not isinstance(self.channel_bars, int) or isinstance(self.channel_bars, bool) or self.channel_bars < 1:
            raise ValueError('channel_bars must be a positive integer')
        if not np.isfinite(self.area_threshold) or not 0.0 < self.area_threshold < 1.0:
            raise ValueError('area_threshold must be between zero and one')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791229374'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(params.channel_bars + 1, 22)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volume_baseline'] = data['volume'].rolling(20, min_periods=20).mean().shift(1)
        out['participation'] = data['volume'].where(data['volume'] > 0) / out['volume_baseline'].where(out['volume_baseline'] > 0)
        out['participation'] = out['participation'].replace([np.inf, -np.inf], np.nan)
        out['lag_return'] = out['return'].shift(1)
        out['lag_participation'] = out['participation'].shift(1)
        out['cross_xy'] = out['lag_participation'] * out['return']
        out['cross_yx'] = out['lag_return'] * out['participation']
        out['signature_area'] = 0.5 * (out['cross_xy'] - out['cross_yx'])
        out['area_scale'] = 0.5 * (out['cross_xy'].abs() + out['cross_yx'].abs())
        out['normalized_area'] = (out['signature_area'] / out['area_scale'].where(out['area_scale'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['upper'] = data['high'].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out['lower'] = data['low'].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        valid = np.isfinite(indicators[['normalized_area', 'upper', 'lower']]).all(axis=1) & np.isfinite(data['close'])
        long_entry = valid & (data['close'] > indicators['upper']) & (indicators['normalized_area'] > params.area_threshold)
        short_entry = valid & (data['close'] < indicators['lower']) & (indicators['normalized_area'] < -params.area_threshold)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, side in enumerate(candidate):
            if position == 0:
                position = int(side)
            elif side == -position:
                position = 0
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': [102.0, 100.0, 98.0, 98.0, 102.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'normalized_area': [0.5, 0.0, -0.5, -0.5, 0.5], 'upper': 101.0, 'lower': 99.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1]
    assert result['size'].gt(0).all()
    derived = strategy.indicators(data, SignatureParams())
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, None, SignatureParams()).data['signal'].eq(0).all()
