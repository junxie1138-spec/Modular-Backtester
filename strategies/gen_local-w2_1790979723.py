from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CohortParams:
    rank_window: int = 4
    tail_threshold: float = 0.8

    def __post_init__(self):
        if not 3 <= self.rank_window <= 5:
            raise ValueError('rank_window must be between 3 and 5')
        if not 0.5 < self.tail_threshold <= 1.0:
            raise ValueError('tail_threshold must be above 0.5 and at most 1.0')


class GeneratedStrategy(BaseStrategy[CohortParams]):
    strategy_id = 'gen_local-w2_1790979723'

    @classmethod
    def params_type(cls):
        return CohortParams

    def warmup_bars(self, params: CohortParams) -> int:
        return 6 + params.rank_window - 1

    def indicators(self, data: pd.DataFrame, params: CohortParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = data['high'] - data['low']
        out['body'] = data['close'] - data['open']
        out['pressure'] = (
            out['body'].div(out['range'].where(out['range'] > 0))
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        )
        out['recent_cohort'] = out['pressure'].rolling(3, min_periods=3).mean()
        out['previous_cohort'] = out['recent_cohort'].shift(3)
        out['migration'] = out['recent_cohort'] - out['previous_cohort']
        rank = out['migration'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='average')
        out['percentile'] = (rank - 1.0) / (params.rank_window - 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CohortParams,
    ) -> SignalFrame:
        long_entry = (
            indicators['migration'].gt(0.0)
            & indicators['percentile'].ge(params.tail_threshold)
        )
        short_entry = (
            indicators['migration'].lt(0.0)
            & indicators['percentile'].le(1.0 - params.tail_threshold)
        )
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = CohortParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'migration': [np.nan, 1, 1, -1, -1, 0, 1, 1],
        'percentile': [np.nan, 1, 0.5, 0, 0, 0.5, 1, 1],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=9)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    prices = pd.DataFrame({
        'open': np.arange(20, dtype=float) + 100,
        'high': np.arange(20, dtype=float) + 102,
        'low': np.arange(20, dtype=float) + 99,
        'close': np.arange(20, dtype=float) + 101,
        'volume': np.full(20, 1000.0),
    }, index=pd.date_range('2020-01-01', periods=20))
    for window in (3, 4, 5):
        p = CohortParams(rank_window=window)
        derived = strategy.indicators(prices, p)
        warmup = strategy.warmup_bars(p)
        assert warmup <= 10
        assert derived['percentile'].iloc[:warmup - 1].isna().all()
        assert derived['percentile'].iloc[warmup - 1:].notna().all()
        prefix = strategy.indicators(prices.iloc[:12], p)
        pd.testing.assert_frame_equal(derived.iloc[:12], prefix)
