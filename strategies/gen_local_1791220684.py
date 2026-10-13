from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrainParams:
    window: int = 18
    history: int = 252
    tail_quantile: float = 0.15
    concentration_quantile: float = 0.65

    def __post_init__(self):
        if self.window < 3 or self.history < 20:
            raise ValueError('window must be >= 3 and history >= 20')
        if not 0.0 < self.tail_quantile < 0.5:
            raise ValueError('tail_quantile must be between 0 and 0.5')
        if not 0.0 < self.concentration_quantile < 1.0:
            raise ValueError('concentration_quantile must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[StrainParams]):
    strategy_id = 'gen_local_1791220684'

    @classmethod
    def params_type(cls):
        return StrainParams

    def warmup_bars(self, params: StrainParams) -> int:
        return int(params.window + params.history + 1)

    def indicators(self, data: pd.DataFrame, params: StrainParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        window = out['return'].rolling(params.window, min_periods=params.window)
        out['net_return'] = window.sum()
        out['largest_return'] = window.max()
        out['smallest_return'] = window.min()
        out['strain'] = out['net_return'] - out['largest_return'] - out['smallest_return']
        absolute = out['return'].abs().rolling(params.window, min_periods=params.window)
        out['absolute_total'] = absolute.sum()
        out['largest_absolute_return'] = absolute.max()
        out['concentration'] = out['largest_absolute_return'] / out['absolute_total'].replace(0.0, np.nan)
        strain_history = out['strain'].shift(1).rolling(params.history, min_periods=params.history)
        out['lower'] = strain_history.quantile(params.tail_quantile)
        out['upper'] = strain_history.quantile(1.0 - params.tail_quantile)
        out['concentration_limit'] = out['concentration'].shift(1).rolling(params.history, min_periods=params.history).quantile(params.concentration_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrainParams) -> SignalFrame:
        strain = indicators['strain']
        distributed = indicators['concentration'].le(indicators['concentration_limit'])
        long_entry = distributed & strain.lt(0.0) & strain.lt(indicators['lower'])
        short_entry = distributed & strain.gt(0.0) & strain.gt(indicators['upper'])
        candidate = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'strain': [np.nan, 2.0, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        'lower': -1.0,
        'upper': 1.0,
        'concentration': 0.2,
        'concentration_limit': 0.5,
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, StrainParams())
    assert result.data['signal'].tolist() == [0, 0, -1, -1, -1, 0, 1, 1]
    assert result.data['size'].gt(0.0).all()
    assert strategy.warmup_bars(StrainParams()) == 271


if __name__ == '__main__':
    _self_check()
