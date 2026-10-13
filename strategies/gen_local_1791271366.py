from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    response_window: int = 4
    peak_window: int = 20
    percentile_window: int = 80
    tail_percentile: float = 0.65

    def __post_init__(self):
        if min(self.response_window, self.peak_window, self.percentile_window) < 2:
            raise ValueError('Window lengths must be at least two.')
        if not 0.5 < self.tail_percentile < 1.0:
            raise ValueError('tail_percentile must be between 0.5 and 1.0.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791271366'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(params.peak_window, params.response_window) + params.percentile_window + 2

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        previous_close = close.shift(1)
        prior_peak = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        drawdown = (1.0 - previous_close / prior_peak).clip(lower=0.0)
        gap = opening / previous_close - 1.0
        intraday_return = close / opening - 1.0
        gap_burden = (-gap.clip(upper=0.0)).rolling(
            params.response_window, min_periods=params.response_window
        ).sum()
        restitution = intraday_return.rolling(
            params.response_window, min_periods=params.response_window
        ).sum()
        recovery = restitution / gap_burden.where(gap_burden > 0.0)
        recovery = recovery.replace([np.inf, -np.inf], np.nan)
        window = params.percentile_window
        quantile = params.tail_percentile
        result = pd.DataFrame(index=data.index)
        result['drawdown'] = drawdown
        result['gap_burden'] = gap_burden
        result['restitution'] = restitution
        result['recovery'] = recovery
        result['drawdown_threshold'] = drawdown.rolling(window, min_periods=window).quantile(quantile).shift(1)
        result['gap_threshold'] = gap_burden.rolling(window, min_periods=window).quantile(quantile).shift(1)
        result['recovery_high'] = recovery.rolling(window, min_periods=window).quantile(quantile).shift(1)
        result['recovery_low'] = recovery.rolling(window, min_periods=window).quantile(1.0 - quantile).shift(1)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        required = ['drawdown', 'gap_burden', 'recovery', 'drawdown_threshold',
                    'gap_threshold', 'recovery_high', 'recovery_low']
        valid = indicators[required].notna().all(axis=1)
        stressed = (
            valid
            & (indicators['drawdown'] > indicators['drawdown_threshold'])
            & (indicators['gap_burden'] > indicators['gap_threshold'])
        )
        long_entry = stressed & (indicators['recovery'] > indicators['recovery_high'])
        short_entry = stressed & (indicators['recovery'] < indicators['recovery_low'])
        df = pd.DataFrame(index=data.index)
        # A position exits to zero exactly when its qualifying entry condition flips false.
        # Four-session memory targets the holding horizon; there is no timed exit.
        df['signal'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    measured = pd.DataFrame({
        'drawdown': 0.10, 'drawdown_threshold': 0.05,
        'gap_burden': 0.04, 'gap_threshold': 0.02,
        'recovery': [2.0, 2.0, 0.5, -1.0, -1.0, 0.5, 2.0],
        'recovery_high': 1.0, 'recovery_low': 0.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    result = strategy.generate_signals(data, measured, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0.0).all()
    measured.loc[:, 'recovery'] = np.nan
    assert strategy.generate_signals(data, measured, None, params).data['signal'].eq(0).all()
