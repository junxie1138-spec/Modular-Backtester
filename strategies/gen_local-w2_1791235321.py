from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockClosureParams:
    topology_window: int = 12
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ShockClosureParams]):
    strategy_id = "gen_local-w2_1791235321"

    @classmethod
    def params_type(cls):
        return ShockClosureParams

    def warmup_bars(self, params: ShockClosureParams) -> int:
        return max(int(params.topology_window) + 1, 15)

    def indicators(self, data: pd.DataFrame, params: ShockClosureParams) -> pd.DataFrame:
        w = int(params.topology_window)
        if w < 8 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("topology_window must be >= 8 and trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data['close']
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        tr = pd.concat([(data['high'] - data['low']), (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        out['return'] = returns
        out['atr'] = tr.rolling(14, min_periods=14).mean()
        out['drawdown'] = close / close.rolling(w, min_periods=w).max() - 1.0
        out['rebound'] = close / close.rolling(w, min_periods=w).min() - 1.0
        for column in ['hub_return', 'hub_age', 'hub_degree', 'current_degree', 'shared_neighbors']:
            out[column] = np.nan
        out['ready'] = False
        if len(data) >= w:
            windows = np.lib.stride_tricks.sliding_window_view(returns.to_numpy(dtype=float), w)
            valid = np.isfinite(windows).all(axis=1)
            magnitude = np.nan_to_num(np.abs(windows), nan=0.0, posinf=0.0, neginf=0.0)
            adjacency = np.zeros((len(windows), w, w), dtype=bool)
            for i in range(w - 1):
                for j in range(i + 1, w):
                    visible = np.ones(len(windows), dtype=bool) if j == i + 1 else magnitude[:, i + 1:j].max(axis=1) < np.minimum(magnitude[:, i], magnitude[:, j])
                    adjacency[:, i, j] = visible
                    adjacency[:, j, i] = visible
            row = np.arange(len(windows))
            hub = magnitude[:, :-1].argmax(axis=1)
            hub_neighbors = adjacency[row, hub, :]
            current_neighbors = adjacency[:, -1, :]
            index = out.index[w - 1:]
            out.loc[index, 'hub_return'] = windows[row, hub]
            out.loc[index, 'hub_age'] = w - 1 - hub
            out.loc[index, 'hub_degree'] = hub_neighbors.sum(axis=1)
            out.loc[index, 'current_degree'] = current_neighbors.sum(axis=1)
            out.loc[index, 'shared_neighbors'] = (hub_neighbors & current_neighbors).sum(axis=1)
            out.loc[index, 'ready'] = valid
        topology = (out['ready'] & out['atr'].gt(0) & out['hub_age'].between(2, 6) & out['hub_degree'].ge(4) & out['current_degree'].ge(3) & out['shared_neighbors'].ge(2) & out['return'].abs().ge(0.65 * out['hub_return'].abs()))
        out['long_setup'] = (topology & out['hub_return'].lt(0) & out['return'].gt(0) & out['drawdown'].lt(-0.005)).fillna(False)
        out['short_setup'] = (topology & out['hub_return'].gt(0) & out['return'].lt(0) & out['rebound'].gt(0.005)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockClosureParams) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        opening = data['open'].to_numpy(dtype=float)
        high = data['high'].to_numpy(dtype=float)
        low = data['low'].to_numpy(dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_setup'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_setup'].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(n):
            desired = decisions[i - 1] if i else 0
            if desired == 0:
                side = 0
                armed = False
                age = 0
            elif side == 0:
                if not (np.isfinite(opening[i]) and opening[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                side = int(desired)
                entry = opening[i]
                stop = side * entry - params.trail_atr * atr[i - 1]
                armed = False
                age = 0
            if not np.isfinite([opening[i], high[i], low[i], close[i]]).all():
                decisions[i] = side
                continue
            if side:
                age += 1
                favorable = high[i] if side == 1 else -low[i]
                adverse = low[i] if side == 1 else -high[i]
                exit_now = adverse <= stop
                if not exit_now:
                    if favorable >= side * entry + 0.0075 * entry:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, favorable - params.trail_atr * atr[i])
                        exit_now = side * close[i] <= stop
                decisions[i] = 0 if exit_now or age >= 4 else side
            elif np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1 if longs[i] else (-1 if shorts[i] else 0)
        df = pd.DataFrame({'signal': decisions, 'size': np.full(n, 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
