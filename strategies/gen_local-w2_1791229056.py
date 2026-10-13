from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEnergyParams:
    window: int = 20
    smoothing: int = 4
    threshold: float = 0.65

    def __post_init__(self):
        if self.window < 2 or self.smoothing < 1:
            raise ValueError('window must be >= 2 and smoothing >= 1')
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[RangeEnergyParams]):
    strategy_id = 'gen_local-w2_1791229056'

    @classmethod
    def params_type(cls):
        return RangeEnergyParams

    @staticmethod
    def warmup_bars(params: RangeEnergyParams) -> int:
        return int(params.window + params.smoothing)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeEnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['energy'] = out['true_range'].pow(2)
        out['weighted_midpoint'] = out['midpoint'] * out['energy']
        out['energy_sum'] = out['energy'].rolling(params.window).sum()
        out['weighted_midpoint_sum'] = out['weighted_midpoint'].rolling(params.window).sum()
        out['centroid'] = out['weighted_midpoint_sum'] / out['energy_sum'].replace(0.0, np.nan)
        out['atr'] = out['true_range'].rolling(params.window).mean()
        out['relative_position'] = (data['close'] - out['centroid']) / out['atr'].replace(0.0, np.nan)
        out['score'] = out['relative_position'].rolling(params.smoothing).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeEnergyParams,
    ) -> SignalFrame:
        score = indicators['score']
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[score.ge(params.threshold)] = 1.0
        decisions.loc[score.le(-params.threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions.ffill().fillna(0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RangeEnergyParams()
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame(index=index)
    scores = pd.DataFrame({'score': [np.nan, 0.7, 0.2, -0.7, -0.2, 0.8, -0.8]}, index=index)
    result = GeneratedStrategy.generate_signals(data, scores, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert result['size'].gt(0).all()
    flat_index = pd.date_range('2020-01-01', periods=50, freq='D')
    flat = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0}, index=flat_index)
    flat_indicators = GeneratedStrategy.indicators(flat, params)
    flat_result = GeneratedStrategy.generate_signals(flat, flat_indicators, None, params).data
    assert flat_result['signal'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.window + params.smoothing
