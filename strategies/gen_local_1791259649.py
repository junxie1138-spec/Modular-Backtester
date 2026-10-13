from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EfficiencyParams:
    rank_window: int = 40
    entry_percentile: float = 0.85

    def __post_init__(self):
        if self.rank_window < 2:
            raise ValueError('rank_window must be at least 2')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[EfficiencyParams]):
    strategy_id = 'gen_local_1791259649'

    @classmethod
    def params_type(cls):
        return EfficiencyParams

    @staticmethod
    def warmup_bars(params: EfficiencyParams) -> int:
        return int(params.rank_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: EfficiencyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['net_progress'] = data['close'].diff()
        out['gap_travel'] = (data['open'] - out['previous_close']).abs()
        out['intraday_range'] = (data['high'] - data['low']).clip(lower=0.0)
        out['total_travel'] = out['gap_travel'] + out['intraday_range']
        out['efficiency'] = (
            out['net_progress'] / out['total_travel'].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['efficiency_percentile'] = out['efficiency'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='average', pct=True)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EfficiencyParams,
    ) -> SignalFrame:
        condition = (
            indicators['efficiency'].gt(0.0)
            & indicators['efficiency_percentile'].ge(params.entry_percentile)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long while the entry condition holds; its reversal exits.
        df['signal'] = condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = EfficiencyParams()
    close = pd.Series([100.0] * 42 + [101.0, 100.0, 100.0],
                      index=pd.date_range('2020-01-01', periods=45))
    opening = close.shift(1).fillna(100.0)
    data = pd.DataFrame({
        'open': opening,
        'high': np.maximum(opening, close) + 0.5,
        'low': np.minimum(opening, close) - 0.5,
        'close': close,
        'volume': 1000000.0,
    })
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result.index.equals(data.index)
    assert result['signal'].iloc[:43].eq(0).all()
    assert result['signal'].iloc[43] == 1
    assert result['signal'].iloc[44] == 0
    assert result['signal'].isin([0, 1]).all()
    assert result['size'].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) == 41


if __name__ == '__main__':
    _self_check()
