from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FrontierParams:
    range_window: int = 18
    percentile_window: int = 126
    entry_quantile: float = 0.65

    def __post_init__(self):
        if self.range_window < 2 or self.percentile_window < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError('entry_quantile must lie strictly between zero and one.')


class GeneratedStrategy(BaseStrategy[FrontierParams]):
    strategy_id = 'gen_local-w2_1791209707'

    @classmethod
    def params_type(cls) -> type[FrontierParams]:
        return FrontierParams

    def warmup_bars(self, params: FrontierParams) -> int:
        return int(params.range_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: FrontierParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        out['range'] = (high - low).where(high >= low)
        out['upper_discovery'] = np.minimum(
            (high - high.shift(1)).clip(lower=0.0), out['range']
        )
        out['lower_discovery'] = np.minimum(
            (low.shift(1) - low).clip(lower=0.0), out['range']
        )
        w = params.range_window
        out['upper_sum'] = out['upper_discovery'].rolling(w, min_periods=w).sum()
        out['lower_sum'] = out['lower_discovery'].rolling(w, min_periods=w).sum()
        out['range_sum'] = out['range'].rolling(w, min_periods=w).sum()
        denominator = out['range_sum'].where(out['range_sum'] > 0.0)
        out['frontier_score'] = (
            (out['upper_sum'] - out['lower_sum']) / denominator
        ).replace([np.inf, -np.inf], np.nan)
        h = params.percentile_window
        out['threshold'] = (
            out['frontier_score'].shift(1)
            .rolling(h, min_periods=h)
            .quantile(params.entry_quantile)
            .clip(lower=0.0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FrontierParams,
    ) -> SignalFrame:
        valid = indicators[['frontier_score', 'threshold']].notna().all(axis=1)
        entry_condition = valid & (
            indicators['frontier_score'] > indicators['threshold']
        )
        df = pd.DataFrame(index=data.index)
        # The position exits only when the same bullish entry predicate flips false.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(1.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    derived = pd.DataFrame({
        'frontier_score': [np.nan, 0.2, 0.3, 0.0, 0.4, -0.1],
        'threshold': [np.nan, 0.1, 0.1, 0.1, 0.1, 0.1],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, FrontierParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(FrontierParams()) == 145
