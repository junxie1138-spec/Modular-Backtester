from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphSeasonParams:
    graph_window: int = 16
    bridge_cutoff: float = 0.24
    history_bars: int = 504
    holding_bars: int = 8
    min_observations: int = 12
    entry_score: float = 1.0
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GraphSeasonParams]):
    strategy_id = 'gen_local-w2_1791326776'

    @classmethod
    def params_type(cls) -> type[GraphSeasonParams]:
        return GraphSeasonParams

    def warmup_bars(self, params: GraphSeasonParams) -> int:
        return int(max(params.history_bars + params.graph_window + params.holding_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GraphSeasonParams) -> pd.DataFrame:
        if params.graph_window < 5 or params.history_bars < 2 or params.holding_bars < 1:
            raise ValueError('Invalid graph, history, or holding window')
        if params.atr_bars < 1 or params.min_observations < 2:
            raise ValueError('Invalid ATR window or observation minimum')
        if params.entry_score <= 0 or params.trail_atr <= 0 or params.breakeven_pct <= 0:
            raise ValueError('Entry and exit parameters must be positive')
        if not 0 < params.bridge_cutoff < 1:
            raise ValueError('Bridge cutoff must lie between zero and one')

        out = pd.DataFrame(index=data.index)
        price = data['close'].where(data['close'] > 0)
        w = params.graph_window
        all_edges = pd.Series(float(w - 1), index=data.index)
        bridges = pd.Series(0.0, index=data.index)
        for lag in range(2, w):
            interior = price.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
            endpoints = pd.concat([price, price.shift(lag)], axis=1).min(axis=1, skipna=False)
            visible = (endpoints > interior).astype(float)
            count = visible.rolling(w - lag, min_periods=w - lag).sum()
            all_edges = all_edges + count
            if lag >= 4:
                bridges = bridges + count
        valid_graph = price.rolling(w, min_periods=w).count().eq(w)
        out['bridge_density'] = (bridges / all_edges).where(valid_graph)
        out['regime'] = (out['bridge_density'] >= params.bridge_cutoff).astype(float).where(valid_graph)

        previous = data['close'].shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1)
        out['true_range'] = ranges.max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['weekday'] = pd.Series(data.index.dayofweek, index=data.index, dtype=float)

        h = params.holding_bars
        entry_open = data['open'].shift(h - 1).where(lambda x: x > 0)
        out['matured_return'] = (price / entry_open - 1.0).replace([np.inf, -np.inf], np.nan)
        out['matured_weekday'] = out['weekday'].shift(h)
        out['matured_regime'] = out['regime'].shift(h)
        out['edge_mean'] = np.nan
        out['edge_std'] = np.nan
        out['observations'] = np.nan
        out['score'] = np.nan

        for weekday in range(5):
            for regime in range(2):
                historical = out['matured_weekday'].eq(weekday) & out['matured_regime'].eq(regime)
                sample = out['matured_return'].where(historical)
                count = sample.rolling(params.history_bars, min_periods=1).count()
                mean = sample.rolling(params.history_bars, min_periods=params.min_observations).mean()
                std = sample.rolling(params.history_bars, min_periods=params.min_observations).std()
                score = mean / std.where(std > 1e-8) * np.sqrt(count)
                current = out['weekday'].eq(weekday) & out['regime'].eq(regime)
                out.loc[current, 'edge_mean'] = mean.loc[current]
                out.loc[current, 'edge_std'] = std.loc[current]
                out.loc[current, 'observations'] = count.loc[current]
                out.loc[current, 'score'] = score.loc[current]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GraphSeasonParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.15, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        scores = indicators['score'].to_numpy(dtype=float)
        direction = 0
        entry = np.nan
        stop = np.nan
        armed = False
        age = 0
        position_size = 0.15
        warmup = self.warmup_bars(params)

        for i in range(n):
            close = closes[i]
            atr = atrs[i]
            if direction:
                sizes[i] = position_size
                if np.isnan(entry):
                    entry = opens[i]
                    entry_atr = atrs[i - 1] if i > 0 else np.nan
                    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(entry_atr) or entry_atr <= 0:
                        direction = 0
                        entry = np.nan
                        continue
                    stop = direction * entry - params.trail_atr * entry_atr
                    age = 0
                    armed = False
                age += 1
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    direction = 0
                    entry = np.nan
                    continue
                signed_close = direction * close
                signed_entry = direction * entry
                if signed_close - signed_entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed-price stops ratchet upward for both directions.
                    # For shorts this corresponds to a decreasing actual stop price.
                    stop = max(stop, signed_entry, signed_close - params.trail_atr * atr)
                if signed_close <= stop or age >= params.holding_bars:
                    direction = 0
                    entry = np.nan
                    continue
                signals[i] = direction
            elif i >= warmup - 1 and np.isfinite(scores[i]) and np.isfinite(atr) and atr > 0 and np.isfinite(close) and close > 0:
                if abs(scores[i]) >= params.entry_score:
                    direction = 1 if scores[i] > 0 else -1
                    position_size = float(np.clip(abs(scores[i]) / 3.0, 0.15, 1.0))
                    entry = np.nan
                    signals[i] = direction
                    sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
