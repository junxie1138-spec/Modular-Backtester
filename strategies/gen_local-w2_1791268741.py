from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    cycle_window: int = 8
    compression_ratio: float = 0.8

    def __post_init__(self):
        if isinstance(self.cycle_window, bool) or not isinstance(self.cycle_window, int) or self.cycle_window < 4:
            raise ValueError('cycle_window must be an integer >= 4')
        if not np.isfinite(self.compression_ratio) or not 0 < self.compression_ratio < 1:
            raise ValueError('compression_ratio must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791268741'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(20, 32 + 2 * params.cycle_window + 3))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(32, dtype=float)
        for k in range(1, 32):
            weights[k] = weights[k - 1] * (k - 1 - 0.5) / k
        # ponytail: fixed 32-tap fractional filter; lengthen if truncation materially changes results.
        fd = close.copy()
        for k in range(1, 32):
            fd = fd + weights[k] * close.shift(k)
        out['fractional_close'] = fd
        out['phase'] = fd.rolling(3, min_periods=3).mean()
        window = 2 * params.cycle_window
        out['phase_center'] = out['phase'].rolling(window, min_periods=window).mean()
        out['phase_scale'] = out['phase'].rolling(window, min_periods=window).std(ddof=0)
        out['excursion'] = ((out['phase'] - out['phase_center']) / out['phase_scale'].where(out['phase_scale'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['rotation'] = out['phase'].diff(2)
        out['relative_range'] = ((high - low) / close).where((high >= low) & (low > 0))
        out['fast_range'] = out['relative_range'].rolling(5, min_periods=5).mean()
        out['slow_range'] = out['relative_range'].rolling(20, min_periods=20).mean()
        out['compression'] = (out['fast_range'] / out['slow_range'].where(out['slow_range'] > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        compressed = indicators['compression'].lt(params.compression_ratio)
        long_entry = compressed & indicators['excursion'].lt(-0.75) & indicators['rotation'].gt(0)
        short_entry = compressed & indicators['excursion'].gt(0.75) & indicators['rotation'].lt(0)
        desired = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicator_check = pd.DataFrame({'compression': [0.5, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5], 'excursion': [-1.0, np.nan, 1.0, 1.0, 0.0, -1.0, 0.0], 'rotation': [1.0, np.nan, -1.0, -1.0, 0.0, 1.0, 0.0]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicator_check, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result.index.equals(data.index) and result['size'].gt(0).all()
    constant = data.reindex(pd.date_range('2020-01-01', periods=100)).ffill()
    derived = strategy.indicators(constant, Params())
    assert not np.isinf(derived.to_numpy()).any()
    assert strategy.generate_signals(constant, derived, None, Params()).data['signal'].eq(0).all()
