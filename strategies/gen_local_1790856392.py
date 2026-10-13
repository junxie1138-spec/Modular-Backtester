from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityRecoveryParams:
    graph_bars: int = 5
    depth_bars: int = 10
    min_depth: float = 0.008
    topology_threshold: float = 0.25

    def __post_init__(self):
        if not isinstance(self.graph_bars, int) or self.graph_bars < 2:
            raise ValueError('graph_bars must be an integer >= 2')
        if not isinstance(self.depth_bars, int) or self.depth_bars < 2:
            raise ValueError('depth_bars must be an integer >= 2')
        if not 0.0 < self.min_depth < 1.0:
            raise ValueError('min_depth must be between zero and one')
        if not 0.0 < self.topology_threshold <= 1.0:
            raise ValueError('topology_threshold must be in (0, 1]')


class GeneratedStrategy(BaseStrategy[VisibilityRecoveryParams]):
    strategy_id = 'gen_local_1790856392'

    @classmethod
    def params_type(cls):
        return VisibilityRecoveryParams

    def warmup_bars(self, params: VisibilityRecoveryParams) -> int:
        return int(max(200, params.graph_bars + 1, params.depth_bars))

    def indicators(self, data: pd.DataFrame, params: VisibilityRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(lambda s: np.isfinite(s) & s.gt(0))
        out['log_close'] = np.log(close)
        out['log_change'] = out['log_close'].diff()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['peak'] = close.rolling(params.depth_bars, min_periods=params.depth_bars).max()
        out['trough'] = close.rolling(params.depth_bars, min_periods=params.depth_bars).min()
        out['drawdown'] = 1.0 - close / out['peak']
        out['drawup'] = close / out['trough'] - 1.0

        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        minimum_slope = pd.Series(np.inf, index=data.index)
        maximum_slope = pd.Series(-np.inf, index=data.index)
        for lag in range(1, params.graph_bars + 1):
            slope = (out['log_close'] - out['log_close'].shift(lag)) / lag
            out[f'slope_{lag}'] = slope
            # Upper visibility requires every intervening point below the chord.
            upper_edge = slope.notna() & slope.lt(minimum_slope)
            lower_edge = slope.notna() & slope.gt(maximum_slope)
            out[f'upper_edge_{lag}'] = upper_edge
            out[f'lower_edge_{lag}'] = lower_edge
            upper += upper_edge.astype(float)
            lower += lower_edge.astype(float)
            minimum_slope = np.minimum(minimum_slope, slope.fillna(np.inf))
            maximum_slope = np.maximum(maximum_slope, slope.fillna(-np.inf))

        out['graph_count'] = close.rolling(params.graph_bars + 1, min_periods=params.graph_bars + 1).count()
        graph_ready = out['graph_count'].eq(params.graph_bars + 1)
        out['upper_degree'] = upper.where(graph_ready)
        out['lower_degree'] = lower.where(graph_ready)
        out['topology'] = (out['upper_degree'] - out['lower_degree']) / (params.graph_bars - 1)
        out['ready'] = graph_ready & out['ma200'].notna() & out['peak'].notna() & out['trough'].notna()
        out['polarity'] = np.select(
            [out['ready'] & out['topology'].ge(params.topology_threshold),
             out['ready'] & out['topology'].le(-params.topology_threshold)],
            [1, -1], default=0
        ).astype(int)
        out['long_entry'] = (
            out['ready'] & out['polarity'].eq(1)
            & close.gt(out['ma200']) & out['log_change'].gt(0)
            & out['drawdown'].ge(params.min_depth)
        )
        out['short_entry'] = (
            out['ready'] & out['polarity'].eq(-1)
            & close.lt(out['ma200']) & out['log_change'].lt(0)
            & out['drawup'].ge(params.min_depth)
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityRecoveryParams) -> SignalFrame:
        polarity = indicators['polarity'].fillna(0).to_numpy(dtype=int)
        long_entry = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            # Neutral topology preserves the position; opposite polarity exits.
            # Regime and depth qualify entries, and never independently exit.
            if position != 0:
                if polarity[i] == -position:
                    position = 0
            elif long_entry[i]:
                position = 1
            elif short_entry[i]:
                position = -1
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = VisibilityRecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=220, freq='D')
    t = np.arange(len(index), dtype=float)
    close = np.exp(4.0 + 0.00001 * t * t)
    data = pd.DataFrame({'open': close, 'high': close * 1.01,
                         'low': close * 0.99, 'close': close,
                         'volume': 1000000.0}, index=index)
    ind = strategy.indicators(data, params)
    assert np.isclose(ind['topology'].iloc[-1], 1.0)
    assert ind['upper_degree'].iloc[-1] == params.graph_bars
    assert ind['lower_degree'].iloc[-1] == 1
    mirrored = data.copy()
    mirrored['close'] = np.exp(4.0 - 0.00001 * t * t)
    assert np.isclose(strategy.indicators(mirrored, params)['topology'].iloc[-1], -1.0)
    assert strategy.warmup_bars(params) >= 200
    probe = pd.DataFrame({'polarity': [1, 0, -1, -1],
                          'long_entry': [True, False, False, False],
                          'short_entry': [False, False, False, False]}, index=index[:4])
    result = strategy.generate_signals(data.iloc[:4], probe, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    assert result['size'].gt(0).all()
    probe['polarity'] = [-1, 0, 1, 1]
    probe['long_entry'] = False
    probe['short_entry'] = [True, False, False, False]
    result = strategy.generate_signals(data.iloc[:4], probe, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
