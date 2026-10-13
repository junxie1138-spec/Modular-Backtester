from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendParams:
    lookback: int = 12
    min_strength: float = 0.15
    min_volume_ratio: float = 1.05
    hold_bars: int = 2

    def __post_init__(self):
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 3:
            raise ValueError('lookback must be an integer >= 3')
        if not np.isfinite(self.min_strength) or not 0 < self.min_strength < 1:
            raise ValueError('min_strength must be between 0 and 1')
        if not np.isfinite(self.min_volume_ratio) or self.min_volume_ratio <= 0:
            raise ValueError('min_volume_ratio must be positive')
        if isinstance(self.hold_bars, bool) or not isinstance(self.hold_bars, int) or self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[TrendParams]):
    strategy_id = 'gen_local_1790970750'

    @classmethod
    def params_type(cls):
        return TrendParams

    def warmup_bars(self, params: TrendParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: TrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = out['return'].rolling(params.lookback, min_periods=params.lookback)
        out['return_sum'] = window.sum()
        out['largest_gain'] = window.max().clip(lower=0)
        out['absolute_sum'] = out['return'].abs().rolling(params.lookback, min_periods=params.lookback).sum()
        out['residual_movement'] = out['absolute_sum'] - out['largest_gain']
        out['strength'] = ((out['return_sum'] - out['largest_gain']) / out['residual_movement'].where(out['residual_movement'] > 1e-12)).replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].astype(float)
        out['valid_volume'] = volume.where(np.isfinite(volume) & volume.ge(0))
        out['prior_volume_mean'] = out['valid_volume'].shift(1).rolling(params.lookback, min_periods=params.lookback).mean()
        out['volume_ratio'] = (out['valid_volume'] / out['prior_volume_mean'].where(out['prior_volume_mean'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['return_agrees'] = out['strength'].ge(params.min_strength) & out['return'].gt(0)
        out['volume_agrees'] = out['volume_ratio'].ge(params.min_volume_ratio)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TrendParams) -> SignalFrame:
        entries = (indicators['return_agrees'].fillna(False).astype(bool) & indicators['volume_agrees'].fillna(False).astype(bool)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(entries):
            if remaining > 0:
                signal[i] = 1
                remaining -= 1
            elif i > 0 and signal[i - 1] == 1:
                # Always emit the scheduled exit, even if entry conditions persist.
                continue
            elif enter:
                signal[i] = 1
                remaining = params.hold_bars - 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    checks = pd.DataFrame({'return_agrees': True, 'volume_agrees': True}, index=index)
    result = strategy.generate_signals(data, checks, None, TrendParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    checks.loc[index[1]:, 'return_agrees'] = False
    result = strategy.generate_signals(data, checks, None, TrendParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    checks['volume_agrees'] = False
    assert strategy.generate_signals(data, checks, None, TrendParams()).data['signal'].eq(0).all()
    calculated = strategy.indicators(data, TrendParams())
    assert strategy.generate_signals(data, calculated, None, TrendParams()).data['signal'].eq(0).all()
