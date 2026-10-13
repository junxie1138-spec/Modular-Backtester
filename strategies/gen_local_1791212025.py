from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    rank_window: int = 40
    recovery_quantile: float = 0.80

    def __post_init__(self):
        if self.peak_window < 2 or self.rank_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < self.recovery_quantile < 1.0:
            raise ValueError('recovery_quantile must be between zero and one.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791212025'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(params.peak_window + params.rank_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['prior_close'] = close.shift(1)
        out['prior_peak'] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max().shift(1)
        out['drawdown_debt'] = (
            out['prior_peak'] - out['prior_close']
        ).clip(lower=0.0)
        out['daily_repayment'] = close.diff()
        fraction = out['daily_repayment'].div(
            out['drawdown_debt'].where(out['drawdown_debt'] > 0.0)
        )
        # Debt-free sessions contribute zero, preserving calendar-sized windows.
        out['repayment_fraction'] = fraction.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).where(out['prior_peak'].notna() & close.notna())
        out['repayment_percentile'] = out['repayment_fraction'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method='average', pct=True)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        entry_condition = (
            indicators['repayment_percentile'].notna()
            & indicators['drawdown_debt'].gt(0.0)
            & indicators['daily_repayment'].gt(0.0)
            & indicators['repayment_percentile'].gt(params.recovery_quantile)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Enter while true; exit exclusively when the same predicate becomes false.
        # The unsmoothed daily predicate targets brief recovery bursts without a time exit.
        df['signal'] = entry_condition.astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame(
        {'open': 100.0, 'high': 101.0, 'low': 99.0,
         'close': 100.0, 'volume': 1000000.0}, index=index
    )
    indicators = pd.DataFrame(
        {'repayment_percentile': [np.nan, 0.95, 0.90, 0.20, 0.95, 0.95],
         'drawdown_debt': [np.nan, 2.0, 1.0, 2.0, 0.0, 2.0],
         'daily_repayment': [np.nan, 1.0, 0.5, -1.0, 1.0, -1.0]},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    assert result['size'].gt(0.0).all()
    longer = pd.concat([data] * 20, ignore_index=True)
    longer.index = pd.date_range('2020-01-01', periods=len(longer), freq='B')
    derived = GeneratedStrategy.indicators(longer, params)
    assert derived.index.equals(longer.index)
    assert derived['repayment_percentile'].first_valid_index() == longer.index[
        GeneratedStrategy.warmup_bars(params) - 1
    ]
    flat = GeneratedStrategy.generate_signals(longer, derived, None, params).data
    assert flat['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
