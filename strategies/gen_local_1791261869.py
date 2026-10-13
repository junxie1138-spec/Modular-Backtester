from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    lookback: int = 60
    lag: int = 4
    threshold: float = 0.25

    def __post_init__(self):
        if self.lookback < 3 or self.lag < 1:
            raise ValueError('lookback must be >= 3 and lag must be >= 1')
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = 'gen_local_1791261869'

    @classmethod
    def params_type(cls) -> type[EchoParams]:
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return params.lookback + params.lag + 1

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['lagged_return'] = out['return'].shift(params.lag)
        out['echo_correlation'] = out['return'].rolling(params.lookback, min_periods=params.lookback).corr(out['lagged_return']).clip(-1.0, 1.0)
        out['block_return'] = out['return'].rolling(params.lag, min_periods=params.lag).sum()
        out['volatility'] = out['return'].rolling(params.lookback, min_periods=params.lookback).std()
        out['forecast'] = out['echo_correlation'] * out['block_return']
        denominator = out['volatility'].where(out['volatility'] > 0) * np.sqrt(params.lag)
        out['score'] = (out['forecast'] / denominator).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EchoParams) -> SignalFrame:
        score = indicators['score']
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[score > params.threshold] = 1.0
        decisions.loc[score < -params.threshold] = 0.0
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions.ffill().fillna(0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = EchoParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'score': [np.nan, 0.8, 0.0, -0.2, -0.8, 0.0, 0.8, -0.8, 0.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 0]
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= params.lookback + params.lag + 1
