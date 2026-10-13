from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 7
    triangle_threshold: float = 0.15
    wick_threshold: float = 0.35
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791326326'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.graph_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 3 <= params.graph_window <= 20:
            raise ValueError('graph_window must be between 3 and 20')
        if params.atr_window < 1 or not 3 <= params.hold_days <= 5:
            raise ValueError('Invalid ATR window or holding horizon')
        if not 0 < params.triangle_threshold <= 1:
            raise ValueError('triangle_threshold must be in (0, 1]')
        if not 0 < params.wick_threshold <= 1 or params.stop_atr <= 0:
            raise ValueError('Invalid wick threshold or stop distance')
        close = data['close'].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        n = params.graph_window
        nodes = [log_close.shift(i) for i in range(n)]
        upper_edges = {}
        lower_edges = {}
        # ponytail: O(window^3) vectorized graph construction; use incremental motifs for large windows.
        for i in range(n):
            for j in range(i + 1, n):
                upper = nodes[i].notna() & nodes[j].notna()
                lower = upper.copy()
                for k in range(i + 1, j):
                    chord = nodes[i] + (nodes[j] - nodes[i]) * ((k - i) / (j - i))
                    upper = upper & (nodes[k] < chord)
                    lower = lower & (nodes[k] > chord)
                upper_edges[i, j] = upper
                lower_edges[i, j] = lower
        upper_triangles = pd.Series(0.0, index=data.index)
        lower_triangles = pd.Series(0.0, index=data.index)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    upper_triangles += (upper_edges[i, j] & upper_edges[i, k] & upper_edges[j, k]).astype(float)
                    lower_triangles += (lower_edges[i, j] & lower_edges[i, k] & lower_edges[j, k]).astype(float)
        graph_valid = log_close.rolling(n, min_periods=n).count().eq(n)
        motif_count = n * (n - 1) * (n - 2) / 6.0
        triangle_balance = ((upper_triangles - lower_triangles) / motif_count).where(graph_valid)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        opening = data['open'].astype(float)
        candle_valid = (high >= pd.concat([opening, close], axis=1).max(axis=1)) & (low <= pd.concat([opening, close], axis=1).min(axis=1)) & (low > 0)
        candle_range = (high - low).where(candle_valid & (high > low))
        lower_wick = pd.concat([opening, close], axis=1).min(axis=1) - low
        upper_wick = high - pd.concat([opening, close], axis=1).max(axis=1)
        wick_balance = ((lower_wick - upper_wick) / candle_range).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
        true_range = true_range.where(candle_valid & previous_close.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = graph_valid & candle_valid & wick_balance.notna() & atr.notna() & (atr > 0)
        return pd.DataFrame({'triangle_balance': triangle_balance, 'wick_balance': wick_balance, 'atr': atr, 'ready': ready}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators['ready'].fillna(False).astype(bool)
        long_entry = ready & (indicators['triangle_balance'] >= params.triangle_threshold) & (indicators['wick_balance'] >= params.wick_threshold)
        short_entry = ready & (indicators['triangle_balance'] <= -params.triangle_threshold) & (indicators['wick_balance'] <= -params.wick_threshold)
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        fixed_stop = np.nan
        held = 0
        for i in range(len(data)):
            exited = False
            if pending_entry:
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    position = 0
                    exited = True
                else:
                    # Entry fills at this open; freeze the prior decision bar's ATR.
                    fixed_stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    held = 0
                pending_entry = False
            if position:
                held += 1
                stopped = np.isfinite(closes[i]) and position * (closes[i] - fixed_stop) < 0
                if stopped or held >= params.hold_days or not np.isfinite(closes[i]):
                    position = 0
                    exited = True
            if position == 0 and not exited and candidates[i] != 0:
                position = int(candidates[i])
                pending_entry = True
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': np.ones(len(data), dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Check one-bar execution delay, fixed stops in both directions, and expiry.
    p = Params()
    index = pd.date_range('2020-01-01', periods=7)
    bars = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 94.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'triangle_balance': 0.0, 'wick_balance': 0.0, 'atr': 1.0, 'ready': True}, index=index)
    features.loc[index[0], ['triangle_balance', 'wick_balance']] = [0.5, 0.5]
    result = GeneratedStrategy.generate_signals(None, bars, features, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    bars.loc[index[2], 'close'] = 95.0
    result = GeneratedStrategy.generate_signals(None, bars, features, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    features.loc[index[0], ['triangle_balance', 'wick_balance']] = [-0.5, -0.5]
    bars.loc[index[2], 'close'] = 105.0
    result = GeneratedStrategy.generate_signals(None, bars, features, None, p).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0, 0, 0]
