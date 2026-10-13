from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    window: int = 6
    topology_gap: float = 0.15
    position_threshold: float = 0.65

    def __post_init__(self):
        if not isinstance(self.window, int) or not 4 <= self.window <= 10:
            raise ValueError('window must be an integer between 4 and 10')
        if not 0 < self.topology_gap <= 1:
            raise ValueError('topology_gap must be in (0, 1]')
        if not 0.5 < self.position_threshold < 1:
            raise ValueError('position_threshold must be in (0.5, 1)')


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = 'gen_local_1791321522'

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        n = params.window
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        values = np.column_stack([close.shift(n - 1 - i).to_numpy() for i in range(n)])
        upper_edges = {}
        lower_edges = {}
        for i in range(n - 1):
            for j in range(i + 1, n):
                finite = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
                upper = finite.copy()
                lower = finite.copy()
                for k in range(i + 1, j):
                    line = values[:, i] + (values[:, j] - values[:, i]) * ((k - i) / (j - i))
                    upper &= values[:, k] < line
                    lower &= values[:, k] > line
                upper_edges[i, j] = upper
                lower_edges[i, j] = lower
        upper_triangles = np.zeros(len(data), dtype=float)
        lower_triangles = np.zeros(len(data), dtype=float)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    upper_triangles += upper_edges[i, j] & upper_edges[i, k] & upper_edges[j, k]
                    lower_triangles += lower_edges[i, j] & lower_edges[i, k] & lower_edges[j, k]
        possible = n * (n - 1) * (n - 2) / 6.0
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        high = high.where(np.isfinite(high) & high.gt(0)).rolling(n, min_periods=n).max()
        low = low.where(np.isfinite(low) & low.gt(0)).rolling(n, min_periods=n).min()
        span = (high - low).where(high.gt(low))
        position = ((close - low) / span).clip(0, 1)
        ready = np.isfinite(values).all(axis=1) & position.notna().to_numpy()
        out = pd.DataFrame(index=data.index)
        out['window_high'] = high
        out['window_low'] = low
        out['window_span'] = span
        out['relative_position'] = position.fillna(0.5)
        out['upper_triangle_density'] = np.where(ready, upper_triangles / possible, 0.0)
        out['lower_triangle_density'] = np.where(ready, lower_triangles / possible, 0.0)
        out['triangle_balance'] = out['upper_triangle_density'] - out['lower_triangle_density']
        out['ready'] = ready
        return out

    @staticmethod
    def _positions(proposals):
        positions = np.zeros(len(proposals), dtype=int)
        held = 0
        for i, proposal in enumerate(proposals):
            if held == 0:
                held = int(proposal)
            elif proposal == -held:
                held = 0
            positions[i] = held
        return positions

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        balance = indicators['triangle_balance']
        position = indicators['relative_position']
        ready = indicators['ready']
        long_entry = ready & balance.ge(params.topology_gap) & position.ge(params.position_threshold)
        short_entry = ready & balance.le(-params.topology_gap) & position.le(1 - params.position_threshold)
        proposals = np.select([long_entry, short_entry], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = self._positions(proposals)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = TopologyParams()
    assert strategy._positions([1, 0, 0, -1, -1, 0, 1, 1]).tolist() == [1, 1, 1, 0, -1, -1, 0, 1]
    close = np.array([9.0, 5.0, 3.0, 3.1, 5.2, 7.5, 8.0, 6.0, 5.0, 7.0])
    data = pd.DataFrame({'open': close, 'high': close + 0.2, 'low': close - 0.2,
                         'close': close, 'volume': 1000.0},
                        index=pd.date_range('2020-01-01', periods=len(close)))
    ind = strategy.indicators(data, params)
    assert ind['triangle_balance'].iloc[5] > params.topology_gap
    assert not ind['ready'].iloc[:5].any()
    signals = strategy.generate_signals(data, ind, None, params).data
    assert signals['signal'].iloc[:6].eq(0).all()
    assert signals['signal'].iloc[6] == 1
    assert signals['signal'].isin([-1, 0, 1]).all() and signals['size'].gt(0).all()
    prefix = strategy.indicators(data.iloc[:8], params)
    pd.testing.assert_frame_equal(prefix, ind.iloc[:8])


if __name__ == '__main__':
    _self_check()
