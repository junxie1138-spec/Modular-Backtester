from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    compression_window: int = 5
    baseline_window: int = 20
    compression_limit: float = 0.75
    gap_threshold: float = 1.4
    hold_bars: int = 8

    def __post_init__(self):
        if not 2 <= self.compression_window <= self.baseline_window:
            raise ValueError('Require 2 <= compression_window <= baseline_window')
        if not 0.0 < self.compression_limit < 1.0:
            raise ValueError('compression_limit must be between zero and one')
        if not np.isfinite(self.gap_threshold) or self.gap_threshold <= 0.0:
            raise ValueError('gap_threshold must be finite and positive')
        if self.hold_bars < 1:
            raise ValueError('hold_bars must be positive')


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = 'gen_local-w2_1791292524'

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return int(max(params.compression_window, params.baseline_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1).where(lambda x: x > 0.0)
        out['gap'] = (data['open'] / out['prior_close'] - 1.0).replace([np.inf, -np.inf], np.nan)
        out['range_fraction'] = ((data['high'] - data['low']) / out['prior_close']).replace([np.inf, -np.inf], np.nan)
        out['fast_range'] = out['range_fraction'].rolling(params.compression_window, min_periods=params.compression_window).mean().shift(1)
        out['baseline_range'] = out['range_fraction'].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(1)
        out['compression'] = out['fast_range'] / out['baseline_range'].where(out['baseline_range'] > 0.0)
        out['gap_squared'] = out['gap'].pow(2)
        out['gap_rms'] = np.sqrt(out['gap_squared'].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(1))
        out['gap_z'] = out['gap'] / out['gap_rms'].clip(lower=0.0001)
        out['release_strength'] = ((out['gap_z'] / params.gap_threshold) * (params.compression_limit / out['compression'].where(out['compression'] > 0.0))).replace([np.inf, -np.inf], np.nan)
        out['entry_size'] = (0.35 + 0.30 * (out['release_strength'].clip(lower=1.0, upper=3.0) - 1.0)).fillna(0.35)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCompressionParams) -> SignalFrame:
        entry = (
            indicators['compression'].gt(0.0)
            & indicators['compression'].le(params.compression_limit)
            & indicators['gap_z'].ge(params.gap_threshold)
            & indicators['release_strength'].notna()
        ).fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators['entry_size'].fillna(0.35).clip(lower=0.35, upper=0.95).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.35, dtype=float)
        exit_bar = -1
        held_size = 0.35
        for i in range(len(data)):
            if i == exit_bar:
                continue
            if i > exit_bar:
                if not entry[i]:
                    continue
                exit_bar = i + params.hold_bars
                held_size = float(entry_sizes[i])
            signals[i] = 1
            sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=11)
    data = pd.DataFrame({name: np.ones(11) for name in ('open', 'high', 'low', 'close', 'volume')}, index=index)
    indicators = pd.DataFrame({'compression': 0.5, 'gap_z': 3.0, 'release_strength': 2.0, 'entry_size': np.linspace(0.35, 0.95, 11)}, index=index)
    params = GapCompressionParams(hold_bars=3)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    assert result['size'].iloc[1:4].nunique() == 1
    assert result['size'].iloc[5] > result['size'].iloc[1]
    indicators['gap_z'] = np.nan
    flat = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert flat['signal'].eq(0).all()
    assert flat['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
