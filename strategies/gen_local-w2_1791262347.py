from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BoundaryParams:
    fast_bars: int = 6
    slow_bars: int = 40
    boundary_bars: int = 8
    compression_ratio: float = 0.75
    asymmetry_threshold: float = 0.25
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[BoundaryParams]):
    strategy_id = 'gen_local-w2_1791262347'

    @classmethod
    def params_type(cls) -> type[BoundaryParams]:
        return BoundaryParams

    def warmup_bars(self, params: BoundaryParams) -> int:
        if not (2 <= params.fast_bars < params.slow_bars):
            raise ValueError('Require 2 <= fast_bars < slow_bars')
        if params.boundary_bars < 2 or params.hold_bars < 1:
            raise ValueError('Invalid boundary or holding window')
        if not (0 < params.compression_ratio < 1):
            raise ValueError('compression_ratio must be between zero and one')
        if not (0 < params.asymmetry_threshold < 1):
            raise ValueError('asymmetry_threshold must be between zero and one')
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError('Require 0 < min_size <= max_size <= 1')
        return int(max(params.slow_bars, params.boundary_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: BoundaryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['daily_range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['daily_range', 'high_gap', 'low_gap']].max(axis=1)
        out['atr_fast'] = out['true_range'].rolling(params.fast_bars).mean()
        out['atr_slow'] = out['true_range'].rolling(params.slow_bars).mean()
        out['high_step'] = data['high'].diff()
        out['low_step'] = data['low'].diff()
        out['high_noise'] = out['high_step'].rolling(params.boundary_bars).std(ddof=0)
        out['low_noise'] = out['low_step'].rolling(params.boundary_bars).std(ddof=0)
        out['noise_sum'] = out['high_noise'] + out['low_noise']
        out['compression'] = out['atr_fast'].div(out['atr_slow'].where(out['atr_slow'] > 0))
        out['asymmetry'] = (out['low_noise'] - out['high_noise']).div(out['noise_sum'].where(out['noise_sum'] > 0))
        out['strength'] = ((out['asymmetry'].abs() - params.asymmetry_threshold) / (1 - params.asymmetry_threshold)).clip(0, 1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BoundaryParams) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (indicators['compression'].lt(params.compression_ratio) & indicators['asymmetry'].abs().ge(params.asymmetry_threshold)).fillna(False)
        candidates = np.where(eligible, np.where(indicators['asymmetry'].gt(0), 1, -1), 0)
        proposed_sizes = (params.min_size + (params.max_size - params.min_size) * indicators['strength'].fillna(0)).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signals[i] = position
                    sizes[i] = entry_size
                continue
            if candidates[i] != 0:
                position = int(candidates[i])
                exit_bar = i + params.hold_bars
                entry_size = float(proposed_sizes[i])
                signals[i] = position
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = BoundaryParams()
    index = pd.date_range('2020-01-01', periods=22)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'compression': 0.5, 'asymmetry': 0.625, 'strength': 0.5}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    expected_size = params.min_size + (params.max_size - params.min_size) * 0.5
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:9].eq(1).all()
    assert result['signal'].iloc[9] == 0
    assert result['signal'].iloc[10:18].eq(1).all()
    assert result['signal'].iloc[18] == 0
    assert np.allclose(result['size'].iloc[1:9], expected_size)
    indicators['asymmetry'] = -0.625
    short_result = strategy.generate_signals(data, indicators, None, params).data
    assert short_result['signal'].iloc[1:9].eq(-1).all()
    assert short_result['signal'].iloc[9] == 0
    indicators['compression'] = np.nan
    flat_result = strategy.generate_signals(data, indicators, None, params).data
    assert flat_result['signal'].eq(0).all()
