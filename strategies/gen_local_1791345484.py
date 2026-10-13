from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverflowParams:
    rank_window: int = 20
    entry_percentile: float = 0.90

    def __post_init__(self):
        if self.rank_window < 3:
            raise ValueError('rank_window must be at least 3')
        if not np.isfinite(self.entry_percentile) or not 0.0 < self.entry_percentile <= 1.0:
            raise ValueError('entry_percentile must be in (0, 1]')


class GeneratedStrategy(BaseStrategy[OverflowParams]):
    strategy_id = 'gen_local_1791345484'

    @classmethod
    def params_type(cls):
        return OverflowParams

    def warmup_bars(self, params: OverflowParams) -> int:
        return int(params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: OverflowParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result['prior_high'] = data['high'].shift(1).rolling(2, min_periods=2).max()
        result['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        result['penetration'] = (data['close'] - result['prior_high']).clip(lower=0.0)
        result['dollar_turnover'] = (data['volume'] * data['close']).where(
            (data['volume'] >= 0.0) & (data['close'] > 0.0)
        )
        result['overflow'] = (
            result['dollar_turnover'] * result['penetration'] / result['range']
        ).replace([np.inf, -np.inf], np.nan)
        result['overflow_percentile'] = result['overflow'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='min', pct=True)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OverflowParams,
    ) -> SignalFrame:
        qualifies = (
            (data['close'] > indicators['prior_high'])
            & (indicators['overflow'] > 0.0)
            & (indicators['overflow_percentile'] >= params.entry_percentile)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Maintaining the entry condition maintains exposure; its reversal exits.
        df['signal'] = qualifies.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OverflowParams(rank_window=3, entry_percentile=0.8)
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    close = pd.Series([10., 10., 10., 10., 12., 11., 12., 11.], index=index)
    data = pd.DataFrame({
        'open': close, 'high': close + 0.5, 'low': close - 0.5,
        'close': close, 'volume': 100.0,
    })
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert (result['size'] > 0.0).all()
    prefix = data.iloc[:6]
    prefix_result = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params
    ).data
    assert prefix_result.equals(result.iloc[:6])
