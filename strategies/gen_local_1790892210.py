from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    window: int = 8
    compression_cutoff: float = 0.5
    direction_cutoff: float = 0.75

    def __post_init__(self):
        if not 2 <= self.window <= 10:
            raise ValueError('window must be between 2 and 10')
        if not 0.0 < self.compression_cutoff <= 1.0:
            raise ValueError('compression_cutoff must be in (0, 1]')
        if not 0.5 < self.direction_cutoff < 1.0:
            raise ValueError('direction_cutoff must be in (0.5, 1)')


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = 'gen_local_1790892210'

    @classmethod
    def params_type(cls) -> type[CapacityParams]:
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        span = (data['high'] - data['low']).where(data['high'] > data['low'])
        volume = data['volume'].where(data['volume'] > 0)
        relative_range = (span / close).replace([np.inf, -np.inf], np.nan)
        capacity_load = (volume / relative_range).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out['relative_range'] = relative_range
        out['capacity_load'] = capacity_load
        out['range_rank'] = relative_range.rolling(params.window, min_periods=params.window).rank(pct=True)
        out['capacity_rank'] = capacity_load.rolling(params.window, min_periods=params.window).rank(pct=True)
        out['close_rank'] = close.rolling(params.window, min_periods=params.window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        eligible = (
            indicators['range_rank'].le(params.compression_cutoff)
            & indicators['capacity_rank'].ge(0.625)
            & indicators[['range_rank', 'capacity_rank', 'close_rank']].notna().all(axis=1)
        )
        entries = np.select(
            [eligible & indicators['close_rank'].ge(params.direction_cutoff),
             eligible & indicators['close_rank'].le(1.0 - params.direction_cutoff)],
            [1, -1],
            default=0,
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(entries):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ranks = pd.DataFrame({'range_rank': 0.25, 'capacity_rank': 0.875, 'close_rank': [1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, ranks, None, CapacityParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    for window in (6, 8, 10):
        params = CapacityParams(window=window)
        values = strategy.indicators(data, params)
        assert values['range_rank'].iloc[:window - 1].isna().all()
        assert strategy.warmup_bars(params) <= 10
