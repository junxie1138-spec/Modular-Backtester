from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendParams:
    lookback: int = 4
    strength_threshold: float = 0.3

    def __post_init__(self):
        if type(self.lookback) is not int or not 2 <= self.lookback <= 9:
            raise ValueError('lookback must be an integer between 2 and 9')
        if not np.isfinite(self.strength_threshold) or self.strength_threshold <= 0.0:
            raise ValueError('strength_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[TrendParams]):
    strategy_id = 'gen_local_1791330726'

    @classmethod
    def params_type(cls):
        return TrendParams

    def warmup_bars(self, params: TrendParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
        n = params.lookback
        previous_close = data['close'].shift(1)
        denominator = previous_close.where(previous_close > 0.0)
        returns = data['close'].div(denominator) - 1.0
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False).div(denominator)
        atr = true_range.rolling(n, min_periods=n).mean()
        return_sum = returns.rolling(n, min_periods=n).sum()
        largest_gain = returns.clip(lower=0.0).rolling(n, min_periods=n).max()
        trimmed_drift = (return_sum - largest_gain) / (n - 1)
        strength = trimmed_drift.div(atr.where(atr > 0.0))
        result = pd.DataFrame({
            'previous_close': previous_close,
            'return': returns,
            'true_range': true_range,
            'atr': atr,
            'return_sum': return_sum,
            'largest_gain': largest_gain,
            'trimmed_drift': trimmed_drift,
            'strength': strength,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TrendParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same predicate controls entry and exit: false immediately exits.
        df['signal'] = indicators['strength'].gt(params.strength_threshold).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = TrendParams()
    close = pd.Series([100.0] * 6 + [101.0, 102.0, 103.0, 104.0, 105.0] + [105.0] * 12,
                      index=pd.date_range('2020-01-01', periods=23))
    data = pd.DataFrame({'open': close, 'high': close + 0.2,
                         'low': close - 0.2, 'close': close, 'volume': 1000.0})
    indicators = GeneratedStrategy.indicators(None, data, params)
    signals = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    expected = indicators['strength'].gt(params.strength_threshold).astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert signals['signal'].eq(1).any() and signals['signal'].iloc[-1] == 0
    assert signals['signal'].iloc[:params.lookback + 1].eq(0).all()
    assert signals['size'].gt(0.0).all()
    spike = pd.Series([100.0] * 6 + [110.0] * 17, index=data.index)
    data['open'] = data['close'] = spike
    data['high'], data['low'] = spike + 0.2, spike - 0.2
    indicators = GeneratedStrategy.indicators(None, data, params)
    assert indicators['strength'].eq(0.0).all()
    assert GeneratedStrategy.warmup_bars(None, params) <= 10


if __name__ == '__main__':
    _self_check()
