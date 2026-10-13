from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 10
    baseline_window: int = 40
    smoothing: int = 5
    compression_ceiling: float = 1.0
    conviction_threshold: float = 0.05

    def __post_init__(self):
        if self.graph_window < 3 or self.baseline_window < self.graph_window:
            raise ValueError('Require baseline_window >= graph_window >= 3')
        if self.smoothing < 1 or not np.isfinite(self.compression_ceiling) or self.compression_ceiling <= 0:
            raise ValueError('Invalid smoothing or compression ceiling')
        if not np.isfinite(self.conviction_threshold) or not 0 < self.conviction_threshold < 1:
            raise ValueError('Require 0 < conviction_threshold < 1')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791251640'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.graph_window, params.baseline_window + 1) + params.smoothing - 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        clean = data[['open', 'high', 'low', 'close', 'volume']].astype(float).replace([np.inf, -np.inf], np.nan)
        valid_bar = clean.notna().all(axis=1) & clean['volume'].ge(0) & clean['high'].ge(clean['low']) & clean['close'].between(clean['low'], clean['high'])
        clean = clean.where(valid_bar)
        spread = clean['high'] - clean['low']
        out['close_acceptance'] = ((2.0 * clean['close'] - clean['high'] - clean['low']) / spread.replace(0.0, np.nan)).where(spread.ne(0.0), 0.0)
        previous_close = clean['close'].shift(1)
        tr = pd.concat([spread, (clean['high'] - previous_close).abs(), (clean['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['true_range'] = tr.where(valid_bar & previous_close.notna())
        out['short_range'] = out['true_range'].rolling(params.graph_window).mean()
        out['baseline_range'] = out['true_range'].rolling(params.baseline_window).mean()
        out['compression_ratio'] = out['short_range'] / out['baseline_range'].replace(0.0, np.nan)
        out['graph_degree_cv'] = np.nan
        out['hub_acceptance'] = np.nan
        width = params.graph_window
        if len(data) >= width:
            volumes = np.lib.stride_tricks.sliding_window_view(clean['volume'].to_numpy(), width)
            acceptance = np.lib.stride_tricks.sliding_window_view(out['close_acceptance'].to_numpy(), width)
            valid = np.isfinite(volumes).all(axis=1) & np.isfinite(acceptance).all(axis=1)
            degrees = np.zeros(volumes.shape, dtype=float)
            # ponytail: O(bars * window^2); use a stack-based graph builder if windows become large.
            for left in range(width - 1):
                barrier = np.full(len(volumes), -np.inf)
                for right in range(left + 1, width):
                    visible = (np.minimum(volumes[:, left], volumes[:, right]) > barrier) & valid
                    degrees[:, left] += visible
                    degrees[:, right] += visible
                    barrier = np.maximum(barrier, volumes[:, right])
            mean_degree = degrees.mean(axis=1)
            concentration = np.divide(degrees.std(axis=1), mean_degree, out=np.zeros(len(volumes)), where=mean_degree > 0)
            weights = degrees * degrees
            weight_sum = weights.sum(axis=1)
            pressure = np.divide((weights * np.nan_to_num(acceptance, nan=0.0)).sum(axis=1), weight_sum, out=np.zeros(len(volumes)), where=weight_sum > 0)
            out.loc[out.index[width - 1:], 'graph_degree_cv'] = np.where(valid, concentration, np.nan)
            out.loc[out.index[width - 1:], 'hub_acceptance'] = np.where(valid, pressure, np.nan)
        out['compression_strength'] = ((params.compression_ceiling - out['compression_ratio']) / 0.25).clip(0.0, 1.0)
        out['topology_strength'] = (out['graph_degree_cv'] / 0.75).clip(0.0, 1.0)
        out['raw_conviction'] = out['compression_strength'] * out['topology_strength'] * out['hub_acceptance']
        out['conviction'] = out['raw_conviction'].rolling(params.smoothing).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        conviction = indicators['conviction'].reindex(data.index).replace([np.inf, -np.inf], np.nan)
        entry_condition = conviction.gt(params.conviction_threshold)
        df = pd.DataFrame(index=data.index)
        # Exposure persists exactly while the smoothed entry condition remains true.
        df['signal'] = entry_condition.astype(int)
        scaled = ((conviction - params.conviction_threshold) / (1.0 - params.conviction_threshold)).clip(0.0, 1.0).fillna(0.0)
        df['size'] = (0.20 + 0.75 * scaled).shift(1).fillna(0.20).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=100)
    t = np.arange(100, dtype=float)
    close = 100.0 + 0.05 * t + np.sin(t / 3.0)
    data = pd.DataFrame({'open': close - 0.1, 'high': close + 0.4, 'low': close - 0.6, 'close': close, 'volume': 1000.0 + 300.0 * np.cos(t)}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:70], params)
    assert np.allclose(full.iloc[:70].to_numpy(), prefix.to_numpy(), equal_nan=True)
    sample = pd.DataFrame({'conviction': [0.0, 0.1, 0.2, 0.0, 0.1, -0.1]}, index=index[:6])
    signals = strategy.generate_signals(data.iloc[:6], sample, None, params).data
    assert signals['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert signals['size'].gt(0).all()
    assert signals['size'].iloc[3] > signals['size'].iloc[2]
    assert strategy.warmup_bars(params) >= params.baseline_window + 1


if __name__ == '__main__':
    _self_check()
