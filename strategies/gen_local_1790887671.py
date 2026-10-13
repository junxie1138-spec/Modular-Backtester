from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 6
    max_cycle_ratio: float = 0.25
    volume_multiple: float = 1.3
    hold_bars: int = 2

    def __post_init__(self):
        if not 3 <= self.lookback <= 9:
            raise ValueError('lookback must be between 3 and 9')
        if not 0.0 <= self.max_cycle_ratio <= 1.0:
            raise ValueError('max_cycle_ratio must be between zero and one')
        if not np.isfinite(self.volume_multiple) or self.volume_multiple < 1.0:
            raise ValueError('volume_multiple must be finite and at least one')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1790887671'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        w = params.lookback
        volume = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        nodes = np.column_stack([volume.shift(age).to_numpy() for age in range(w, 0, -1)])
        valid = np.isfinite(nodes).all(axis=1) & (nodes > 0).all(axis=1)
        extra_edges = np.zeros(len(data), dtype=float)
        # Adjacent nodes always connect; each additional edge adds one independent cycle.
        for left in range(w - 2):
            for right in range(left + 2, w):
                obstruction = nodes[:, left + 1:right].max(axis=1)
                visible = obstruction < np.minimum(nodes[:, left], nodes[:, right])
                extra_edges += visible & valid

        result = pd.DataFrame(index=data.index)
        result['cycle_ratio'] = np.where(valid, extra_edges / (w - 2), np.nan)
        result['volume_decay'] = valid & (nodes[:, -1] < nodes[:, 0])
        result['prior_volume_mean'] = volume.shift(1).rolling(w, min_periods=w).mean()
        result['prior_high'] = data['high'].shift(1).rolling(w, min_periods=w).max()
        result['prior_low'] = data['low'].shift(1).rolling(w, min_periods=w).min()
        result['volume_surge'] = (volume >= params.volume_multiple * result['prior_volume_mean']).fillna(False)
        result['breakout'] = np.select(
            [data['close'] > result['prior_high'], data['close'] < result['prior_low']],
            [1, -1], default=0,
        ).astype(int)
        result['eligible'] = (
            result['cycle_ratio'].le(params.max_cycle_ratio)
            & result['volume_decay']
            & result['volume_surge']
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        entries = indicators['breakout'].where(indicators['eligible'].fillna(False), 0).fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, entry in enumerate(entries):
            if position and i >= exit_at:
                position = 0
            elif not position and entry:
                position = int(entry)
                exit_at = i + params.hold_bars
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({
        'open': [100.0] * 12,
        'high': [101.0] * 7 + [103.0] * 5,
        'low': [99.0] * 12,
        'close': [100.0] * 7 + [102.0] * 5,
        'volume': [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 200.0, 40.0, 40.0, 40.0, 40.0],
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived['cycle_ratio'].iloc[7] == 0.0
    assert derived['eligible'].iloc[7]
    assert strategy.warmup_bars(params) <= 10
    for hold in (1, 2):
        for direction in (1, -1):
            test_params = VisibilityParams(hold_bars=hold)
            forced = pd.DataFrame({'breakout': direction, 'eligible': True}, index=index)
            signals = strategy.generate_signals(data, forced, None, test_params).data
            assert signals['signal'].iloc[0] == 0
            assert signals['signal'].iloc[1:hold + 1].eq(direction).all()
            assert signals['signal'].iloc[hold + 1] == 0
            assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
