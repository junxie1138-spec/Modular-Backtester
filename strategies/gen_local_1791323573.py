from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRetirementParams:
    peak_window: int = 63
    recovery_window: int = 18
    history_window: int = 252
    breakout_percentile: float = 0.65

    def __post_init__(self):
        if min(self.peak_window, self.recovery_window, self.history_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < self.breakout_percentile < 1.0:
            raise ValueError('breakout_percentile must lie strictly between zero and one.')


class GeneratedStrategy(BaseStrategy[DrawdownRetirementParams]):
    strategy_id = 'gen_local_1791323573'

    @classmethod
    def params_type(cls):
        return DrawdownRetirementParams

    def warmup_bars(self, params: DrawdownRetirementParams) -> int:
        return int(params.peak_window + 2 * params.recovery_window + params.history_window + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownRetirementParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        result['peak'] = data['close'].rolling(params.peak_window, min_periods=params.peak_window).max()
        result['depth'] = (1.0 - data['close'] / result['peak'].where(result['peak'] > 0.0)).clip(lower=0.0)
        result['burden'] = result['depth'].rolling(params.recovery_window, min_periods=params.recovery_window).mean()
        result['retirement'] = result['burden'].shift(params.recovery_window) - result['burden']
        result['threshold'] = result['retirement'].rolling(params.history_window, min_periods=params.history_window).quantile(params.breakout_percentile).shift(1)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownRetirementParams) -> SignalFrame:
        ready = indicators[['retirement', 'threshold']].notna().all(axis=1)
        # Remain long only while the original percentile breakout condition holds.
        condition = ready & indicators['retirement'].gt(indicators['threshold']) & indicators['retirement'].gt(0.0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = DrawdownRetirementParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    n = np.arange(700)
    close = 100.0 + np.where(n % 120 < 60, n % 120, 120 - n % 120)
    data = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000000.0}, index=pd.date_range('2020-01-01', periods=len(n), freq='B'))
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    expected = (derived['retirement'].gt(derived['threshold']) & derived['retirement'].gt(0.0)).astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert signals['signal'].isin([0, 1]).all() and signals['size'].gt(0.0).all()
    assert signals['signal'].eq(1).any()
    assert ((signals['signal'].shift(1) == 1) & signals['signal'].eq(0)).any()
    prefix = data.iloc[:560]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:560], prefix_signals)


if __name__ == '__main__':
    _self_check()
