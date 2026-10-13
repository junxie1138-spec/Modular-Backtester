from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    slope_window: int = 10
    model_window: int = 63
    percentile_window: int = 126
    entry_percentile: float = 0.85
    holding_bars: int = 4

    def __post_init__(self):
        for value in (self.slope_window, self.model_window, self.percentile_window):
            if isinstance(value, bool) or not isinstance(value, int) or value < 3:
                raise ValueError('Lookback windows must be integers >= 3')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one')
        if isinstance(self.holding_bars, bool) or not isinstance(self.holding_bars, int) or not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be an integer from 3 to 5')


def _fixed_hold(candidates, holding_bars):
    signals = np.zeros(len(candidates), dtype=int)
    position = 0
    exit_bar = -1
    for i, candidate in enumerate(candidates):
        if position and i == exit_bar:
            position = 0
        elif not position and candidate:
            position = int(candidate)
            exit_bar = i + holding_bars
        signals[i] = position
    return signals


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = 'gen_local-w2_1791345853'

    @classmethod
    def params_type(cls):
        return TrendInnovationParams

    @staticmethod
    def warmup_bars(params):
        return int(params.slope_window + params.model_window + params.percentile_window + 2)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0)).replace([np.inf, -np.inf], np.nan)
        length = params.slope_window
        weights = np.arange(length, dtype=float) - (length - 1) / 2.0
        weights /= np.dot(weights, weights)
        slope = np.full(len(data), np.nan)
        if len(data) >= length:
            slope[length - 1:] = np.convolve(out['log_close'].to_numpy(), weights[::-1], mode='valid')
        out['slope'] = slope
        out['lagged_slope'] = out['slope'].shift(1)
        window = params.model_window
        current = out['slope'].rolling(window, min_periods=window)
        lagged = out['lagged_slope'].rolling(window, min_periods=window)
        out['slope_mean'] = current.mean()
        out['lagged_mean'] = lagged.mean()
        out['slope_covariance'] = current.cov(out['lagged_slope'], ddof=0)
        out['lagged_variance'] = lagged.var(ddof=0)
        out['persistence'] = (out['slope_covariance'] / out['lagged_variance'].where(out['lagged_variance'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['intercept'] = out['slope_mean'] - out['persistence'] * out['lagged_mean']
        out['forecast'] = out['intercept'].shift(1) + out['persistence'].shift(1) * out['lagged_slope']
        out['innovation'] = out['slope'] - out['forecast']
        out['absolute_innovation'] = out['innovation'].abs()
        out['entry_threshold'] = out['absolute_innovation'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        ready = indicators[['slope', 'innovation', 'entry_threshold']].notna().all(axis=1)
        extreme = indicators['absolute_innovation'] > indicators['entry_threshold']
        long_entry = ready & extreme & (indicators['slope'] > 0) & (indicators['innovation'] > 0)
        short_entry = ready & extreme & (indicators['slope'] < 0) & (indicators['innovation'] < 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_hold(candidates, params.holding_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    candidates = np.array([1] + [-1] * 9)
    raw = _fixed_hold(candidates, 4)
    assert raw.tolist() == [1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    shifted = pd.Series(raw).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1]
    assert not _fixed_hold(np.zeros(10, dtype=int), 4).any()
