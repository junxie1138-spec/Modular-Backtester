from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFanParams:
    graph_window: int = 6
    evidence_bars: int = 3
    entry_threshold: float = 0.12
    min_size: float = 0.20
    max_size: float = 0.95

    def __post_init__(self):
        if self.graph_window < 3 or self.evidence_bars < 1:
            raise ValueError('Invalid lookback')
        if not 0.0 < self.entry_threshold <= 1.0:
            raise ValueError('Invalid threshold')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Invalid sizing bounds')


class GeneratedStrategy(BaseStrategy[GapFanParams]):
    strategy_id = 'gen_local_1790863157'

    @classmethod
    def params_type(cls):
        return GapFanParams

    def warmup_bars(self, params):
        return int(max(20, params.graph_window) + params.evidence_bars)

    def indicators(self, data, params):
        clean = data[['open', 'close']].astype(float).replace([np.inf, -np.inf], np.nan)
        opening = clean['open'].where(clean['open'] > 0)
        close = clean['close'].where(clean['close'] > 0)
        gap = opening / close.shift(1) - 1.0
        magnitude = gap.abs()
        intraday = close / opening - 1.0
        scale = magnitude.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
        edges = {}
        for lag in range(1, params.graph_window):
            if lag == 1:
                visible = magnitude.notna() & magnitude.shift(1).notna()
            else:
                blockers = magnitude.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
                visible = (magnitude > blockers) & (magnitude.shift(lag) > blockers)
            edges[lag] = visible.fillna(False)
        degree = pd.Series(0.0, index=data.index)
        triangles = pd.Series(0.0, index=data.index)
        for edge in edges.values():
            degree += edge.astype(float)
        for newer in range(1, params.graph_window - 1):
            for older in range(newer + 1, params.graph_window):
                closed = edges[newer] & edges[older] & edges[older - newer].shift(newer, fill_value=False)
                triangles += closed.astype(float)
        possible = (degree * (degree - 1.0) / 2.0).replace(0.0, np.nan)
        clustering = (triangles / possible).fillna(0.0).clip(0.0, 1.0)
        fan = degree / (params.graph_window - 1.0) * (1.0 - 0.5 * clustering)
        rejection = (-np.sign(gap) * intraday / magnitude.replace(0.0, np.nan)).clip(0.0, 1.0)
        amplitude = (magnitude / scale).clip(0.0, 3.0) / 3.0
        raw = np.sign(gap) * (1.0 - 2.0 * rejection) * amplitude * fan
        valid = magnitude.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        raw = raw.where(valid & intraday.notna()).replace([np.inf, -np.inf], np.nan)
        score = raw.rolling(params.evidence_bars, min_periods=params.evidence_bars).mean().fillna(0.0)
        return pd.DataFrame({
            'gap': gap.fillna(0.0),
            'intraday': intraday.fillna(0.0),
            'gap_scale': scale.fillna(0.0),
            'degree': degree,
            'triangles': triangles,
            'clustering': clustering,
            'fan': fan,
            'rejection': rejection.fillna(0.0),
            'amplitude': amplitude.fillna(0.0),
            'raw_pressure': raw.fillna(0.0),
            'score': score,
        }, index=data.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def generate_signals(self, data, indicators, ctx, params):
        score = indicators['score'].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        desired = np.where(score >= params.entry_threshold, 1, np.where(score <= -params.entry_threshold, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        entry_size = params.min_size
        for i, candidate in enumerate(desired):
            if position and candidate == -position:
                position = 0
                entry_size = params.min_size
            elif position == 0 and candidate:
                position = int(candidate)
                confidence = min(abs(float(score[i])) / (2.0 * params.entry_threshold), 1.0)
                entry_size = params.min_size + (params.max_size - params.min_size) * confidence
            signals[i] = position
            sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = GapFanParams()
    index = pd.date_range('2020-01-01', periods=80)
    rng = np.random.default_rng(7)
    opening = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    close = opening * np.exp(rng.normal(0.0, 0.006, len(index)))
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.01,
                         'low': np.minimum(opening, close) * 0.99, 'close': close,
                         'volume': 1000000.0}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:50], strategy.indicators(data.iloc[:50], params))
    assert np.isfinite(full.to_numpy()).all()
    assert full['score'].iloc[:strategy.warmup_bars(params) - 1].eq(0.0).all()
    sample = data.iloc[:7]
    controlled = pd.DataFrame({'score': [0.15, 0.0, -0.3, -0.3, 0.0, 0.2, 0.2]}, index=sample.index)
    result = strategy.generate_signals(sample, controlled, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0.0).all()
    assert result['size'].iloc[1] == result['size'].iloc[2] < result['size'].iloc[4]
