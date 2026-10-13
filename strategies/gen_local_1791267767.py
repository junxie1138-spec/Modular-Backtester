from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AllocationTideParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    baseline_window: int = 40
    seasonal_days: int = 7
    strength_threshold: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must lie strictly between zero and one')
        if self.fractional_window < 2 or self.baseline_window < 2:
            raise ValueError('lookback windows must be at least two')
        if not 5 <= self.seasonal_days <= 8:
            raise ValueError('seasonal_days must be between five and eight')
        if not np.isfinite(self.strength_threshold):
            raise ValueError('strength_threshold must be finite')


class GeneratedStrategy(BaseStrategy[AllocationTideParams]):
    strategy_id = 'gen_local_1791267767'

    @classmethod
    def params_type(cls):
        return AllocationTideParams

    def warmup_bars(self, params: AllocationTideParams) -> int:
        return int(params.fractional_window + params.baseline_window - 1)

    def indicators(self, data: pd.DataFrame, params: AllocationTideParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional filter; increase fractional_window for longer memory.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        mean = fractional_close.rolling(params.baseline_window).mean()
        std = fractional_close.rolling(params.baseline_window).std(ddof=0)
        strength = ((fractional_close - mean) / std.where(std.gt(0.0))).replace([np.inf, -np.inf], np.nan)
        day = data.index.day
        seasonal = ((day >= 1) & (day <= params.seasonal_days)) | ((day >= 16) & (day <= 15 + params.seasonal_days))
        return pd.DataFrame({
            'fractional_close': fractional_close,
            'fractional_mean': mean,
            'fractional_std': std,
            'strength': strength,
            'seasonal': seasonal,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AllocationTideParams) -> SignalFrame:
        eligible = (
            indicators['seasonal'].fillna(False).astype(bool)
            & indicators['strength'].notna()
            & indicators['strength'].gt(params.strength_threshold)
        ).fillna(False)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        # Confirmation persists while eligible; its first reversal exits immediately.
        df['signal'] = confirmed.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AllocationTideParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=9, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({
        'seasonal': [True, True, True, True, True, True, True, True, False],
        'strength': [np.nan, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert result['size'].gt(0.0).all()
    longer_index = pd.date_range('2020-01-01', periods=100, freq='B')
    longer_data = data.reindex(longer_index)
    longer_data['close'] = np.arange(100, dtype=float) + 100.0
    derived = strategy.indicators(longer_data, params)
    assert derived['strength'].first_valid_index() == longer_index[strategy.warmup_bars(params) - 1]
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
