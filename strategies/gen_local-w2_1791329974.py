from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    window: int = 9
    min_strength: float = 0.15

    def __post_init__(self):
        if not 2 <= self.window <= 9:
            raise ValueError('window must be between 2 and 9')
        if not np.isfinite(self.min_strength) or not 0.0 < self.min_strength < 1.0:
            raise ValueError('min_strength must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = 'gen_local-w2_1791329974'

    @classmethod
    def params_type(cls):
        return RangeMigrationParams

    @staticmethod
    def warmup_bars(params: RangeMigrationParams) -> int:
        return params.window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data['high'].replace([np.inf, -np.inf], np.nan)
        low = data['low'].replace([np.inf, -np.inf], np.nan)
        valid = high.ge(low) & high.notna() & low.notna()
        out['high_move'] = high.where(valid).diff()
        out['low_move'] = low.where(valid).diff()
        up = np.minimum(out['high_move'].clip(lower=0), out['low_move'].clip(lower=0))
        down = np.minimum((-out['high_move']).clip(lower=0), (-out['low_move']).clip(lower=0))
        out['translation'] = up - down
        out['boundary_churn'] = out['high_move'].abs() + out['low_move'].abs()
        out['translation_sum'] = out['translation'].rolling(params.window, min_periods=params.window).sum()
        out['churn_sum'] = out['boundary_churn'].rolling(params.window, min_periods=params.window).sum()
        out['strength'] = (2.0 * out['translation_sum'] / out['churn_sum'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeMigrationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same predicate maintains exposure; its reversal is the only exit.
        df['signal'] = indicators['strength'].gt(params.min_strength).fillna(False).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RangeMigrationParams()
    center = np.concatenate((np.arange(100.0, 125.0), np.arange(123.0, 98.0, -1.0)))
    data = pd.DataFrame({'open': center, 'high': center + 1.0, 'low': center - 1.0, 'close': center, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=len(center)))
    indicators = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    expected = indicators['strength'].gt(params.min_strength).astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert signals['signal'].iloc[:params.window + 1].eq(0).all()
    assert signals['signal'].iloc[params.window + 1] == 1
    assert signals['signal'].iloc[-1] == 0
    assert signals['signal'].isin([0, 1]).all() and signals['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
