from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 12
    baseline_window: int = 40
    shock_threshold: float = 2.0
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.graph_window < 3 or self.baseline_window < 3:
            raise ValueError('Windows must be at least three bars.')
        if not np.isfinite(self.shock_threshold) or self.shock_threshold <= 0:
            raise ValueError('shock_threshold must be positive and finite.')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five.')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = 'gen_local-w2_1790969567'

    @classmethod
    def params_type(cls):
        return TopologyParams

    @staticmethod
    def warmup_bars(params):
        return int(params.graph_window + params.baseline_window + 1)

    @staticmethod
    def indicators(data, params):
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        current = returns.to_numpy()
        upper_degree = np.zeros(len(data), dtype=float)
        lower_degree = np.zeros(len(data), dtype=float)
        interior_max = np.full(len(data), -np.inf)
        interior_min = np.full(len(data), np.inf)

        # Horizontal visibility requires every intervening vertex to lie
        # strictly below both endpoints; invert returns for the lower graph.
        for lag in range(1, params.graph_window):
            previous = returns.shift(lag).to_numpy()
            finite = np.isfinite(current) & np.isfinite(previous)
            upper_degree += finite & (interior_max < np.minimum(current, previous))
            lower_degree += finite & (interior_min > np.maximum(current, previous))
            interior_max = np.maximum(interior_max, previous)
            interior_min = np.minimum(interior_min, previous)

        ready = returns.rolling(params.graph_window).count().eq(params.graph_window)
        upper = pd.Series(upper_degree, index=data.index).where(ready)
        lower = pd.Series(lower_degree, index=data.index).where(ready)
        balance = (upper - lower) / (upper + lower).replace(0, np.nan)
        historical = balance.shift(1).rolling(params.baseline_window)
        mean = historical.mean()
        std = historical.std(ddof=0)
        shock = ((balance - mean) / std.where(std > 1e-12)).replace([np.inf, -np.inf], np.nan)
        # Only a fresh excursion triggers entry, not a persistent regime.
        event = shock.abs().ge(params.shock_threshold) & shock.shift(1).abs().lt(params.shock_threshold)
        return pd.DataFrame({
            'return': returns,
            'upper_degree': upper,
            'lower_degree': lower,
            'degree_balance': balance,
            'baseline_mean': mean,
            'baseline_std': std,
            'topology_shock': shock,
            'entry_event': event.fillna(False),
        }, index=data.index)

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        shock = indicators['topology_shock'].to_numpy(dtype=float)
        events = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if direction != 0 and i == exit_bar:
                direction = 0
                # Emit a flat target on expiry even if a new event occurs.
                continue
            if direction == 0 and events[i] and np.isfinite(shock[i]):
                direction = int(np.sign(shock[i]))
                if direction != 0:
                    exit_bar = i + params.hold_bars
                    strength = float(np.clip(abs(shock[i]) / (3.0 * params.shock_threshold), 0.0, 1.0))
                    entry_size = params.min_size + (params.max_size - params.min_size) * strength
            if direction != 0:
                signals[i] = direction
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = TopologyParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'topology_shock': [3.0, -6.0, 0.0, 0.0, 6.0, -6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'entry_event': [True, True, False, False, True, True, True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert result['size'].iloc[1:5].nunique() == 1
    assert result['size'].iloc[6:10].nunique() == 1
    assert result['size'].iloc[6] > result['size'].iloc[1] > 0
    flat = pd.DataFrame({'close': np.full(100, 100.0)}, index=pd.date_range('2021-01-01', periods=100))
    derived = GeneratedStrategy.indicators(flat, params)
    assert not derived['entry_event'].any()


if __name__ == '__main__':
    _self_check()
