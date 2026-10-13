from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphCompressionParams:
    graph_window: int = 12
    compression_ratio: float = 0.8
    consensus_threshold: float = 0.6
    min_bridge_span: float = 2.5
    holding_bars: int = 18

    def __post_init__(self):
        if self.graph_window < 6 or self.holding_bars < 1:
            raise ValueError('graph_window must be >= 6 and holding_bars >= 1')
        if not 0.0 < self.compression_ratio < 1.0:
            raise ValueError('compression_ratio must be between zero and one')
        if not 0.0 < self.consensus_threshold <= 1.0:
            raise ValueError('consensus_threshold must be in (0, 1]')
        if not np.isfinite(self.min_bridge_span) or self.min_bridge_span < 2.0:
            raise ValueError('min_bridge_span must be finite and >= 2')


class GeneratedStrategy(BaseStrategy[GraphCompressionParams]):
    strategy_id = 'gen_local_1791250630'

    @classmethod
    def params_type(cls):
        return GraphCompressionParams

    def warmup_bars(self, params: GraphCompressionParams) -> int:
        return int(params.graph_window + 1)

    def indicators(self, data: pd.DataFrame, params: GraphCompressionParams) -> pd.DataFrame:
        w = params.graph_window
        split = w // 2
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        candle_range = ((data['high'] - data['low']) / close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        candle_range = candle_range.where(candle_range >= 0)
        # Columns run chronologically from oldest to newest; every row is causal.
        ranges = np.column_stack([candle_range.shift(w - 1 - k).to_numpy() for k in range(w)])
        prices = np.column_stack([close.shift(w - 1 - k).to_numpy() for k in range(w)])
        heights = -ranges
        valid = np.isfinite(ranges).all(axis=1) & np.isfinite(prices).all(axis=1)
        count = np.zeros(len(data), dtype=float)
        span_sum = np.zeros(len(data), dtype=float)
        signed_sum = np.zeros(len(data), dtype=float)
        # Horizontal visibility: all interior heights must lie below both endpoints.
        # ponytail: O(rows * window^2) edge enumeration; use a stack graph algorithm for large windows.
        for i in range(split):
            interior_max = np.full(len(data), -np.inf)
            for j in range(i + 1, w):
                if j > i + 1:
                    interior_max = np.maximum(interior_max, heights[:, j - 1])
                if j < split or j - i < 2:
                    continue
                visible = valid & (np.minimum(heights[:, i], heights[:, j]) > interior_max)
                span = j - i
                vote = np.sign(prices[:, j] - prices[:, i])
                count += visible.astype(float)
                span_sum += visible.astype(float) * span
                signed_sum += np.where(visible, vote * span, 0.0)
        old_mean = ranges[:, :split].mean(axis=1)
        new_mean = ranges[:, split:].mean(axis=1)
        compression = np.divide(new_mean, old_mean, out=np.full(len(data), np.nan), where=valid & (old_mean > 0))
        mean_span = np.divide(span_sum, count, out=np.zeros(len(data)), where=count > 0)
        consensus = np.divide(signed_sum, span_sum, out=np.zeros(len(data)), where=span_sum > 0)
        return pd.DataFrame({
            'relative_range': candle_range,
            'compression': compression,
            'bridge_count': np.where(valid, count, np.nan),
            'bridge_span': np.where(valid, mean_span, np.nan),
            'bridge_consensus': np.where(valid, consensus, np.nan),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GraphCompressionParams) -> SignalFrame:
        eligible = (
            indicators['compression'].le(params.compression_ratio)
            & indicators['bridge_count'].ge(2)
            & indicators['bridge_span'].ge(params.min_bridge_span)
        ).fillna(False)
        score = indicators['bridge_consensus'].fillna(0.0)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[eligible & score.ge(params.consensus_threshold)] = 1
        candidate.loc[eligible & score.le(-params.consensus_threshold)] = -1
        confirmed = candidate.where(candidate.ne(0) & candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entered = -1
        for t in range(len(data)):
            if position != 0 and t - entered >= params.holding_bars:
                position = 0
            elif position == 0 and confirmed[t] != 0:
                position = int(confirmed[t])
                entered = t
            signal[t] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GraphCompressionParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=50)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    assert strategy.generate_signals(data, strategy.indicators(data, params), None, params).data['signal'].eq(0).all()
    for side in (1, -1):
        synthetic = pd.DataFrame({'compression': 0.5, 'bridge_count': 3.0, 'bridge_span': 4.0, 'bridge_consensus': 0.0}, index=index)
        synthetic.iloc[:2, synthetic.columns.get_loc('bridge_consensus')] = side
        synthetic.iloc[5:7, synthetic.columns.get_loc('bridge_consensus')] = -side
        result = strategy.generate_signals(data, synthetic, None, params).data
        assert result['signal'].iloc[:2].eq(0).all()
        assert result['signal'].iloc[2:20].eq(side).all()
        assert result['signal'].iloc[20:].eq(0).all()
        assert result['signal'].ne(0).sum() == params.holding_bars
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
