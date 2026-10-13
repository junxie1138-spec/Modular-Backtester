from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeDominanceParams:
    reference_bars: int = 18
    smoothing_bars: int = 4
    dominance_threshold: float = 0.60

    def __post_init__(self):
        if self.reference_bars < 1 or self.smoothing_bars < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 0.5 < self.dominance_threshold < 1.0:
            raise ValueError('Dominance threshold must lie between 0.5 and 1.')


class GeneratedStrategy(BaseStrategy[RangeDominanceParams]):
    strategy_id = 'gen_local_1791335811'

    @classmethod
    def params_type(cls):
        return RangeDominanceParams

    def warmup_bars(self, params: RangeDominanceParams) -> int:
        return int(params.reference_bars + params.smoothing_bars)

    def indicators(self, data: pd.DataFrame, params: RangeDominanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['lag_low'] = data['low'].shift(params.reference_bars)
        out['lag_high'] = data['high'].shift(params.reference_bars)
        out['range_width'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['lag_width'] = (out['lag_high'] - out['lag_low']).where(out['lag_high'] > out['lag_low'])
        # Integrate the lagged interval's CDF over the current interval.
        out['daily_dominance'] = (
            (data['high'] - out['lag_low']).clip(lower=0).pow(2)
            - (data['low'] - out['lag_low']).clip(lower=0).pow(2)
            - (data['high'] - out['lag_high']).clip(lower=0).pow(2)
            + (data['low'] - out['lag_high']).clip(lower=0).pow(2)
        ).div(2.0 * out['range_width'] * out['lag_width']).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
        out['dominance'] = out['daily_dominance'].rolling(
            params.smoothing_bars, min_periods=params.smoothing_bars
        ).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeDominanceParams) -> SignalFrame:
        bullish = indicators['dominance'].gt(params.dominance_threshold).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Two consecutive bullish closes enter; the first false condition exits.
        df['signal'] = (bullish & bullish.shift(1, fill_value=False)).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RangeDominanceParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'dominance': [0.7, 0.7, 0.4, 0.7, 0.7, 0.7, 0.4, np.nan]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 1, 1, 0]
    assert (result['size'] > 0).all()
