from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 16
    baseline_window: int = 20
    recent_window: int = 5
    topology_threshold: float = 0.8
    return_threshold: float = 1.0
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.graph_window, self.baseline_window, self.recent_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if self.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')
        if not (np.isfinite(self.topology_threshold) and self.topology_threshold > 0):
            raise ValueError('topology_threshold must be finite and positive.')
        if not (np.isfinite(self.return_threshold) and self.return_threshold > 0):
            raise ValueError('return_threshold must be finite and positive.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791228393'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.graph_window + params.baseline_window + 1,
            params.recent_window + params.baseline_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        returns = np.log(close).diff()
        interior_max = pd.Series(-np.inf, index=data.index)
        edge_span = pd.Series(0.0, index=data.index)
        signed_span = pd.Series(0.0, index=data.index)

        # A horizontal visibility edge requires both endpoints above every interior close.
        for lag in range(1, params.graph_window + 1):
            past = close.shift(lag)
            visible = close.notna() & past.notna() & (np.minimum(close, past) > interior_max)
            weight = visible.astype(float) * lag
            edge_span += weight
            signed_span += weight * np.sign(close - past).fillna(0.0)
            interior_max = pd.concat([interior_max, past], axis=1).max(axis=1)

        complete = close.rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        edge_span = edge_span.where(complete)
        signed_span = signed_span.where(complete)
        topology = signed_span / edge_span.replace(0.0, np.nan)
        topology_baseline = topology.shift(1).rolling(params.baseline_window).mean()
        topology_jump = topology - topology_baseline

        recent_mean = returns.rolling(params.recent_window).mean()
        prior_returns = returns.shift(params.recent_window)
        prior_mean = prior_returns.rolling(params.baseline_window).mean()
        prior_std = prior_returns.rolling(params.baseline_window).std(ddof=1)
        shift_scale = prior_std * np.sqrt(1.0 / params.recent_window + 1.0 / params.baseline_window)
        return_shift = (recent_mean - prior_mean) / shift_scale.replace(0.0, np.nan)

        return pd.DataFrame({
            'returns': returns,
            'edge_span': edge_span,
            'signed_span': signed_span,
            'topology': topology,
            'topology_baseline': topology_baseline,
            'topology_jump': topology_jump,
            'recent_mean': recent_mean,
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'shift_scale': shift_scale,
            'return_shift': return_shift,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        topology_jump = indicators['topology_jump']
        return_shift = indicators['return_shift']
        ready = np.isfinite(topology_jump) & np.isfinite(return_shift)
        long_entry = ready & (topology_jump >= params.topology_threshold) & (return_shift >= params.return_threshold)
        short_entry = ready & (topology_jump <= -params.topology_threshold) & (return_shift <= -params.return_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered = -params.hold_bars

        for i, entry in enumerate(entries):
            if position and i - entered >= params.hold_bars:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
                entered = i
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        # Entry fills one bar later; exit fills exactly hold_bars after that entry.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params(hold_bars=3)
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    agreement = np.array([1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=float)
    indicators = pd.DataFrame({
        'topology_jump': agreement,
        'return_shift': agreement * 2.0,
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0, 1]
    assert result['size'].gt(0).all()
    indicators['return_shift'] = -agreement * 2.0
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()
