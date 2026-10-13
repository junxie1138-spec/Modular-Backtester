from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    fractional_order: float = 0.45
    fractional_window: int = 48
    recent_window: int = 5
    baseline_window: int = 30
    mean_threshold: float = 0.6
    polarity_threshold: float = 0.35
    holding_bars: int = 18
    min_size: float = 0.35
    max_size: float = 0.95

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(self.fractional_window, self.recent_window, self.baseline_window) < 2:
            raise ValueError('Indicator windows must be at least two')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        if not np.isfinite(self.mean_threshold) or self.mean_threshold <= 0:
            raise ValueError('mean_threshold must be finite and positive')
        if not 0.0 < self.polarity_threshold < 1.0:
            raise ValueError('polarity_threshold must be between zero and one')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = 'gen_local_1791293005'

    @classmethod
    def params_type(cls):
        return ChangepointParams

    def warmup_bars(self, params: ChangepointParams) -> int:
        return int(params.fractional_window + params.recent_window + params.baseline_window + 1)

    def indicators(self, data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: fixed-length fractional filter; increase the window if truncation materially changes results.
        fractional_close = sum(weights[k] * close.shift(k) for k in range(params.fractional_window))
        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fractional_close
        out['fractional_step'] = fractional_close.diff()
        out['total_energy'] = out['fractional_step'].pow(2)
        out['downside_energy'] = out['fractional_step'].clip(upper=0).pow(2)
        r = params.recent_window
        b = params.baseline_window
        out['recent_energy'] = out['total_energy'].rolling(r, min_periods=r).sum()
        out['recent_downside'] = out['downside_energy'].rolling(r, min_periods=r).sum()
        out['baseline_energy'] = out['total_energy'].shift(r).rolling(b, min_periods=b).sum()
        out['baseline_downside'] = out['downside_energy'].shift(r).rolling(b, min_periods=b).sum()
        out['recent_down_share'] = out['recent_downside'] / out['recent_energy'].replace(0, np.nan)
        out['baseline_down_share'] = out['baseline_downside'] / out['baseline_energy'].replace(0, np.nan)
        out['recent_mean'] = fractional_close.rolling(r, min_periods=r).mean()
        out['baseline_mean'] = fractional_close.shift(r).rolling(b, min_periods=b).mean()
        out['baseline_std'] = fractional_close.shift(r).rolling(b, min_periods=b).std(ddof=0)
        out['mean_shift'] = (out['recent_mean'] - out['baseline_mean']) / out['baseline_std'].replace(0, np.nan)
        out['polarity_change'] = out['baseline_down_share'] - out['recent_down_share']
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            out['mean_shift'].ge(params.mean_threshold)
            & out['polarity_change'].ge(params.polarity_threshold)
            & out['baseline_down_share'].ge(0.5)
            & out['recent_down_share'].lt(0.5)
        ).fillna(False)
        mean_excess = out['mean_shift'] / params.mean_threshold - 1.0
        polarity_excess = out['polarity_change'] / params.polarity_threshold - 1.0
        out['strength'] = ((mean_excess + polarity_excess) / 2.0).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangepointParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0 and i >= exit_bar:
                exit_bar = -1
                continue
            if exit_bar < 0:
                if not entries[i]:
                    continue
                exit_bar = i + params.holding_bars
                held_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
            signals[i] = 1
            sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = ChangepointParams()
    strategy = GeneratedStrategy()
    count = 180
    index = pd.date_range('2020-01-01', periods=count, freq='B')
    close = 100.0 + np.arange(count) * 0.1 + np.sin(np.arange(count))
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000000.0}, index=index)
    actual = strategy.indicators(data, params)
    assert actual['fractional_close'].first_valid_index() == index[params.fractional_window - 1]
    altered = data.copy()
    altered.loc[index[120]:, 'close'] *= 2.0
    pd.testing.assert_frame_equal(actual.iloc[:120], strategy.indicators(altered, params).iloc[:120])
    forced = pd.DataFrame({'entry': True, 'strength': np.linspace(0.0, 1.0, count)}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, forced, ctx, params).data
    h = params.holding_bars
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:h + 1].eq(1).all()
    assert result['signal'].iloc[h + 1] == 0
    assert result['signal'].iloc[h + 2] == 1
    assert result['size'].iloc[1:h + 1].eq(params.min_size).all()
    assert result['size'].iloc[h + 2] > params.min_size
    assert result['signal'].isin([0, 1]).all()
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
