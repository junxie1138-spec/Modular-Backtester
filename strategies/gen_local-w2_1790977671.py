from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TerritoryParams:
    range_window: int = 8
    pressure_window: int = 2
    rank_window: int = 4
    compression_percentile: float = 0.75
    pressure_percentile: float = 0.75

    def __post_init__(self):
        if not 2 <= self.range_window <= 10:
            raise ValueError('range_window must be between 2 and 10')
        if self.pressure_window < 1 or self.rank_window < 2:
            raise ValueError('Invalid pressure or rank window')
        if self.pressure_window + self.rank_window > 10:
            raise ValueError('Combined pressure warmup exceeds 10 bars')
        if not 0 < self.compression_percentile <= 1:
            raise ValueError('Invalid compression percentile')
        if not 0.5 < self.pressure_percentile <= 1:
            raise ValueError('Invalid pressure percentile')


class GeneratedStrategy(BaseStrategy[TerritoryParams]):
    strategy_id = 'gen_local-w2_1790977671'

    @classmethod
    def params_type(cls):
        return TerritoryParams

    def warmup_bars(self, params: TerritoryParams) -> int:
        return int(max(params.range_window, params.pressure_window + params.rank_window))

    def indicators(self, data: pd.DataFrame, params: TerritoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        out['relative_range'] = out['range'] / data['close'].where(data['close'] > 0)
        out['range_rank'] = out['relative_range'].rolling(params.range_window).rank(pct=True)
        out['fresh_up'] = data['high'].diff().clip(lower=0)
        out['fresh_down'] = (-data['low'].diff()).clip(lower=0)
        out['territory_balance'] = (
            (out['fresh_up'] - out['fresh_down']) / out['range'].replace(0, np.nan)
        ).where(out['range'].ne(0), 0.0)
        out['territory_balance'] = out['territory_balance'].replace([np.inf, -np.inf], np.nan)
        out['pressure'] = out['territory_balance'].rolling(params.pressure_window).sum()
        out['up_rank'] = out['pressure'].rolling(params.rank_window).rank(pct=True)
        out['down_rank'] = out['pressure'].rolling(params.rank_window).rank(ascending=False, pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TerritoryParams,
    ) -> SignalFrame:
        compressed = indicators['range_rank'].le(params.compression_percentile)
        long_entry = (
            compressed
            & indicators['pressure'].gt(0)
            & indicators['up_rank'].ge(params.pressure_percentile)
        ).fillna(False)
        short_entry = (
            compressed
            & indicators['pressure'].lt(0)
            & indicators['down_rank'].ge(params.pressure_percentile)
        ).fillna(False)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        position = np.zeros(len(data), dtype=int)
        state = 0
        for i, direction in enumerate(candidate):
            if state == 0:
                state = int(direction)
            elif direction == -state:
                state = 0
            position[i] = state
        df = pd.DataFrame({'signal': position, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = TerritoryParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['range_rank'].iloc[:7].isna().all()
    direction = np.array([1, 0, 0, -1, -1, 0, 1, 1])
    fixture = pd.DataFrame({
        'range_rank': 0.25,
        'pressure': direction.astype(float),
        'up_rank': np.where(direction == 1, 1.0, 0.25),
        'down_rank': np.where(direction == -1, 1.0, 0.25),
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=8)
    signals = strategy.generate_signals(data, fixture, ctx, params).data
    assert signals['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert signals['size'].gt(0).all()
    for window in (6, 8, 10):
        for pressure in (1, 2, 3):
            assert strategy.warmup_bars(TerritoryParams(range_window=window, pressure_window=pressure)) <= 10
