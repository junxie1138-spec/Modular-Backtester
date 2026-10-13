from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BrakingParams:
    peak_window: int = 6
    brake_fraction: float = 0.5
    min_drawdown: float = 0.005

    def __post_init__(self):
        if not 2 <= self.peak_window <= 10:
            raise ValueError('peak_window must be between 2 and 10')
        if not 0.0 < self.brake_fraction < 1.0:
            raise ValueError('brake_fraction must be between zero and one')
        if not 0.0 < self.min_drawdown < 1.0:
            raise ValueError('min_drawdown must be between zero and one')


class GeneratedStrategy(BaseStrategy[BrakingParams]):
    strategy_id = 'gen_local-w2_1790984621'

    @classmethod
    def params_type(cls) -> type[BrakingParams]:
        return BrakingParams

    @staticmethod
    def warmup_bars(params: BrakingParams) -> int:
        return max(params.peak_window, 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: BrakingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['roc'] = close.div(close.shift(1)).sub(1.0)
        out['previous_roc'] = out['roc'].shift(1)
        out['acceleration'] = out['roc'].sub(out['previous_roc'])
        out['braking'] = out['acceleration'].div(out['previous_roc'].abs().replace(0.0, np.nan))
        out['range_fraction'] = data['high'].sub(data['low']).div(close)
        out['previous_range'] = out['range_fraction'].shift(1)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - close.div(out['peak'])
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BrakingParams) -> SignalFrame:
        braking_wave = (
            indicators['roc'].lt(0.0)
            & indicators['previous_roc'].lt(0.0)
            & indicators['braking'].ge(params.brake_fraction)
            & indicators['drawdown'].ge(params.min_drawdown)
            & indicators['range_fraction'].gt(0.0)
            & indicators['range_fraction'].lt(indicators['previous_range'])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The same predicate controls entry and reversal exit; no timed exit.
        df['signal'] = braking_wave.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = BrakingParams()
    close = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.6, 95.45, 95.7, 95.4])
    widths = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0, 2.0, 2.0])
    data = pd.DataFrame({'open': close, 'high': close + widths / 2.0, 'low': close - widths / 2.0, 'close': close, 'volume': 1000000.0}, index=pd.date_range('2020-01-01', periods=len(close)))
    indicators = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0]
    assert result['size'].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
