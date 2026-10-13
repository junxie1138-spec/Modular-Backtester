from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    strength_window: int = 60
    strength_threshold: float = 0.8
    channel_window: int = 8
    reference_window: int = 40
    compression_ratio: float = 0.75
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791267950'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(params.fractional_window + params.strength_window,
                       params.channel_window + 1,
                       params.reference_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.fractional_window, params.strength_window,
               params.channel_window, params.reference_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('Fractional order must be between zero and one.')
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError('Compression ratio must be between zero and one.')
        if not np.isfinite(params.strength_threshold) or params.strength_threshold <= 0:
            raise ValueError('Strength threshold must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = out['log_close'].copy()
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * out['log_close'].shift(k)
        out['fractional_close'] = fractional_close
        out['fractional_mean'] = fractional_close.rolling(params.strength_window).mean().shift(1)
        out['fractional_std'] = fractional_close.rolling(params.strength_window).std(ddof=0).shift(1)
        out['fractional_strength'] = ((fractional_close - out['fractional_mean']) /
                                      out['fractional_std'].replace(0.0, np.nan))
        out['relative_range'] = ((data['high'] - data['low']).clip(lower=0) /
                                 data['close'].where(data['close'] > 0))
        out['short_range'] = out['relative_range'].rolling(params.channel_window).mean().shift(1)
        out['reference_range'] = out['relative_range'].rolling(params.reference_window).mean().shift(1)
        out['range_ratio'] = out['short_range'] / out['reference_range'].replace(0.0, np.nan)
        out['channel_high'] = data['high'].rolling(params.channel_window).max().shift(1)
        out['channel_low'] = data['low'].rolling(params.channel_window).min().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['compressed'] = out['range_ratio'].le(params.compression_ratio).fillna(False)
        out['channel_direction'] = np.select(
            [data['close'].gt(out['channel_high']), data['close'].lt(out['channel_low'])],
            [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('Holding period must be positive.')
        strength = indicators['fractional_strength']
        compressed = indicators['compressed'].fillna(False).astype(bool)
        direction = indicators['channel_direction'].fillna(0)
        entries = np.select(
            [compressed & direction.eq(1) & strength.gt(params.strength_threshold),
             compressed & direction.eq(-1) & strength.lt(-params.strength_threshold)],
            [1, -1], default=0).astype(int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0:
                if i - entry_bar == params.hold_bars:
                    position = 0
                targets[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                entry_bar = i
            targets[i] = position
        df = pd.DataFrame({'signal': targets, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    sample = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    derived = pd.DataFrame({
        'fractional_strength': [2.0] + [-2.0] * 11,
        'compressed': [True] * 12,
        'channel_direction': [1] + [-1] * 11,
    }, index=index)
    result = GeneratedStrategy.generate_signals(sample, derived, None, CompressionParams())
    assert result.data['signal'].tolist() == [0] + [1] * 8 + [0] + [-1] * 2
    assert result.data['size'].gt(0).all()
