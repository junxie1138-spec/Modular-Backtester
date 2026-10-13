from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    volatility_window: int = 8
    smoothing_window: int = 3
    percentile_window: int = 126
    entry_percentile: float = 0.65

    def __post_init__(self):
        if self.volatility_window < 2 or self.smoothing_window < 1 or self.percentile_window < 2:
            raise ValueError('Invalid rolling window')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one')


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = 'gen_local_1791275540'

    @classmethod
    def params_type(cls):
        return MigrationParams

    def warmup_bars(self, params: MigrationParams) -> int:
        return int(params.volatility_window + params.smoothing_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        opening = data['open'].where(data['open'] > 0)
        closing = data['close'].where(data['close'] > 0)
        result['session_return'] = np.log(closing / opening)
        result['overnight_return'] = np.log(opening / closing.shift(1))
        result['session_volatility'] = result['session_return'].rolling(params.volatility_window).std(ddof=0)
        result['overnight_volatility'] = result['overnight_return'].rolling(params.volatility_window).std(ddof=0)
        denominator = result['overnight_volatility'].replace(0.0, np.nan)
        result['volatility_ratio'] = result['session_volatility'] / denominator
        result['migration'] = result['volatility_ratio'].rolling(params.smoothing_window).mean()
        result['threshold'] = result['migration'].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        result['session_momentum'] = result['session_return'].rolling(params.volatility_window).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        required = ['migration', 'threshold', 'session_momentum']
        ready = indicators[required].notna().all(axis=1)
        entry = ready & indicators['migration'].gt(indicators['threshold']) & indicators['session_momentum'].gt(0.0)
        df = pd.DataFrame(index=data.index)
        # Qualification reversal is the sole exit: retain exposure exactly while entry remains true.
        df['signal'] = entry.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = MigrationParams()
    strategy = GeneratedStrategy()
    rng = np.random.default_rng(19)
    count = 400
    overnight = rng.normal(0.0, 0.002, count)
    session = rng.normal(0.001, 0.005, count)
    close = 100.0 * np.exp(np.cumsum(overnight + session))
    previous = np.r_[100.0, close[:-1]]
    opening = previous * np.exp(overnight)
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.001, 'low': np.minimum(opening, close) * 0.999, 'close': close, 'volume': np.full(count, 1000000.0)}, index=pd.date_range('2020-01-01', periods=count, freq='B'))
    derived = strategy.indicators(data, params)
    actual = strategy.generate_signals(data, derived, None, params).data
    expected = ((derived['migration'] > derived['threshold']) & (derived['session_momentum'] > 0.0)).astype(int).shift(1).fillna(0).astype(int)
    assert actual['signal'].equals(expected)
    assert actual['signal'].isin([0, 1]).all() and actual['signal'].eq(1).any()
    assert actual['size'].gt(0.0).all()
    prefix = data.iloc[:300]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(actual.iloc[:300], prefix_signals)
