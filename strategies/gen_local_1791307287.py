from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_window: int = 5
    smoothing: int = 3
    volatility_window: int = 20
    entry_threshold: float = 0.10
    min_size: float = 0.20
    max_size: float = 0.95

    def __post_init__(self):
        if self.roc_window < 1 or self.smoothing < 1 or self.volatility_window < 2:
            raise ValueError('Invalid lookback lengths')
        if not np.isfinite(self.entry_threshold) or self.entry_threshold < 0:
            raise ValueError('entry_threshold must be finite and nonnegative')
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = 'gen_local_1791307287'

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(3 * params.roc_window + params.smoothing,
                       params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        w = params.roc_window
        close = data['close'].astype(float).where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        out['daily_return'] = close.pct_change(fill_method=None)
        out['fast_roc'] = close.pct_change(periods=w, fill_method=None)
        out['slow_roc'] = close.pct_change(periods=2 * w, fill_method=None)
        out['fast_acceleration'] = out['fast_roc'].diff(w)
        out['slow_acceleration'] = out['slow_roc'].diff(w)
        out['fast_smoothed'] = out['fast_acceleration'].rolling(
            params.smoothing, min_periods=params.smoothing).mean()
        out['slow_smoothed'] = out['slow_acceleration'].rolling(
            params.smoothing, min_periods=params.smoothing).mean()
        out['volatility'] = out['daily_return'].rolling(
            params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        scale = out['volatility'].where(out['volatility'] > 1e-12)
        out['fast_strength'] = out['fast_smoothed'] / (scale * np.sqrt(w))
        out['slow_strength'] = out['slow_smoothed'] / (scale * np.sqrt(2 * w))
        out['score'] = np.minimum(out['fast_strength'], out['slow_strength'])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        score = indicators['score'].replace([np.inf, -np.inf], np.nan)
        entry = ((indicators['slow_roc'] > 0) &
                 (score > params.entry_threshold)).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The position remains long precisely while the entry condition holds.
        # Its first false bar reverses the condition and schedules an exit.
        df['signal'] = entry.astype(int)
        strength = (score - params.entry_threshold).clip(lower=0).fillna(0)
        desired_size = params.min_size + (params.max_size - params.min_size) * (
            strength / (1.0 + strength))
        df['size'] = desired_size.shift(1).fillna(params.min_size).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = AccelerationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({name: np.ones(6) for name in
                         ('open', 'high', 'low', 'close', 'volume')}, index=index)
    derived = pd.DataFrame({'slow_roc': [0.1] * 6,
                            'score': [np.nan, 0.2, 1.0, -0.1, 0.3, 0.4]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert result['size'].between(params.min_size, params.max_size).all()
    assert result['size'].iloc[3] > result['size'].iloc[2]
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data['signal'].eq(0).all()
    assert strategy.warmup_bars(params) >= 3 * params.roc_window + params.smoothing
