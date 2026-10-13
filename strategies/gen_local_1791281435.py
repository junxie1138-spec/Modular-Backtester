from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    window: int = 6
    smoothing: int = 3
    pressure_threshold: float = 0.06

    def __post_init__(self):
        if not 3 <= self.window <= 6:
            raise ValueError('window must be between 3 and 6')
        if not 1 <= self.smoothing <= 3:
            raise ValueError('smoothing must be between 1 and 3')
        if not np.isfinite(self.pressure_threshold) or self.pressure_threshold < 0:
            raise ValueError('pressure_threshold must be finite and nonnegative')


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = 'gen_local_1791281435'

    @classmethod
    def params_type(cls):
        return ElasticParams

    @staticmethod
    def warmup_bars(params: ElasticParams) -> int:
        return int(params.window + params.smoothing + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['lagged_return'] = out['return'].shift(1)
        out['volatility'] = out['return'].rolling(params.window, min_periods=params.window).std(ddof=0)
        out['autocorrelation'] = out['return'].rolling(params.window, min_periods=params.window).corr(out['lagged_return']).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out['normalized_shock'] = out['return'].div(out['volatility'].where(out['volatility'] > 0))
        out['rebound_forecast'] = out['normalized_shock'] * out['autocorrelation'].clip(upper=0.0)
        out['pressure'] = out['rebound_forecast'].rolling(params.smoothing, min_periods=params.smoothing).mean().replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The identical condition controls entry and reversal exit; no timed exit.
        df['signal'] = (indicators['pressure'].notna() & (indicators['pressure'] > params.pressure_threshold)).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def self_check():
    params = ElasticParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'pressure': [np.nan, 0.1, 0.2, -0.1, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert (result['size'] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) == 10
    flat = pd.DataFrame({name: np.ones(20) for name in ['open', 'high', 'low', 'close', 'volume']}, index=pd.date_range('2020-01-01', periods=20))
    flat_indicators = GeneratedStrategy.indicators(flat, params)
    assert GeneratedStrategy.generate_signals(flat, flat_indicators, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    self_check()
