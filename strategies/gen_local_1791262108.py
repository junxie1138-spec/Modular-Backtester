from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphParams:
    graph_window: int = 12
    volatility_window: int = 20
    chord_threshold: float = 0.55
    shock_threshold: float = 0.75
    profit_target: float = 0.008
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[GraphParams]):
    strategy_id = 'gen_local_1791262108'

    @classmethod
    def params_type(cls) -> type[GraphParams]:
        return GraphParams

    @staticmethod
    def warmup_bars(params: GraphParams) -> int:
        return int(max(200, params.graph_window, params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.volatility_window < 2:
            raise ValueError('Indicator windows are too short')
        if not 0.0 < params.chord_threshold < 1.0:
            raise ValueError('chord_threshold must be between zero and one')
        if not np.isfinite(params.shock_threshold) or params.shock_threshold <= 0.0:
            raise ValueError('shock_threshold must be positive and finite')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError('profit_target must be positive and finite')
        if params.holding_bars not in (1, 2):
            raise ValueError('holding_bars must be one or two')

        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        n = params.graph_window
        chords = pd.Series(0.0, index=data.index)
        # Horizontal visibility: every intervening close is below both endpoints.
        # Count only edges whose endpoints lie within the current n-bar graph.
        for distance in range(2, n):
            interior_max = close.shift(1).rolling(distance - 1).max()
            visible = close.gt(interior_max) & close.shift(distance).gt(interior_max)
            chords = chords + visible.astype(float).rolling(n - distance).sum()
        complete = close.rolling(n).count().eq(n)
        # A horizontal visibility graph has at most n-2 nonadjacent edges.
        density = (chords / float(n - 2)).where(complete)
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(params.volatility_window).std(ddof=0)
        shock = (returns / volatility.where(volatility.gt(0.0))).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            'ma200': close.rolling(200).mean(),
            'chord_density': density,
            'return': returns,
            'volatility': volatility,
            'shock': shock,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GraphParams,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        density = indicators['chord_density'].to_numpy(dtype=float)
        shock = indicators['shock'].to_numpy(dtype=float)
        valid = (
            np.isfinite(close) & (close > 0.0) & np.isfinite(ma)
            & np.isfinite(density) & np.isfinite(shock)
        )
        # Sparse graphs follow shocks; chord-rich graphs fade shocks.
        forecast = np.where(density < params.chord_threshold, shock, -shock)
        candidates = np.zeros(len(data), dtype=int)
        candidates[valid & (close > ma) & (forecast >= params.shock_threshold)] = 1
        candidates[valid & (close < ma) & (forecast <= -params.shock_threshold)] = -1

        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            # Yesterday's close decision becomes today's position at the open.
            previous = int(decisions[i - 1]) if i else 0
            if position == 0 and previous != 0:
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    raise ValueError('Entry open must be positive and finite')
                position = previous
                entry_price = opens[i]
                entry_bar = i
            elif previous == 0:
                position = 0

            if position != 0:
                gain = position * (close[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.holding_bars
                # Close-observed exits execute at the following open.
                decisions[i] = 0 if target_hit or time_hit else position
            else:
                decisions[i] = candidates[i]

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 102.0, 'low': 98.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        'ma200': 90.0, 'chord_density': 0.0, 'shock': 1.0,
    }, index=index)
    p = GraphParams(profit_target=0.5)
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1]
    p.profit_target = 0.005
    data['close'] = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 0, 1]
    ind['ma200'] = 110.0
    ind['shock'] = -1.0
    data['close'] = [100.0, 99.0, 100.0, 99.0, 100.0, 99.0]
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result['signal'].tolist() == [0, -1, 0, -1, 0, -1]
    trend = pd.DataFrame({'close': np.arange(1.0, 211.0)}, index=pd.date_range('2020-01-01', periods=210))
    topology = GeneratedStrategy.indicators(trend, GraphParams())
    assert topology['chord_density'].iloc[-1] == 0.0
    assert GeneratedStrategy.warmup_bars(GraphParams(volatility_window=250)) == 251


if __name__ == '__main__':
    _self_check()
