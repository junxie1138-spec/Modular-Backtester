from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 12
    hub_degree: int = 4
    slope_days: int = 5
    atr_window: int = 14
    stop_atr: float = 1.2
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = 'gen_local_1791246644'

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(max(200 + params.slope_days, params.graph_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.graph_window < 3 or not 2 <= params.hub_degree <= params.graph_window:
            raise ValueError('Invalid visibility window or hub degree')
        if params.slope_days < 1 or params.atr_window < 1:
            raise ValueError('Indicator windows must be positive')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0 or params.hold_days not in (1, 2):
            raise ValueError('Stop must be positive and holding period must be one or two days')
        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['ma_slope'] = out['ma200'].diff(params.slope_days)
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        true_range = true_range.where(np.isfinite(true_range) & (true_range >= 0))
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        degree = pd.Series(0.0, index=data.index)
        ceiling = pd.Series(-np.inf, index=data.index)
        # Edge to lag j exists iff every intervening close is below both endpoints.
        for lag in range(1, params.graph_window):
            older = close.shift(lag)
            visible = ceiling < np.minimum(close, older)
            degree += (visible & close.notna() & older.notna()).astype(float)
            ceiling = pd.Series(np.maximum(ceiling, older), index=data.index)
        valid = close.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        out['degree'] = degree.where(valid)
        out['previous_degree'] = out['degree'].shift(1)
        out['degree_change'] = out['degree'] - out['previous_degree']
        out['close_change'] = close.diff()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TopologyParams) -> SignalFrame:
        if params.hold_days not in (1, 2) or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Invalid stop or holding period')
        ready = pd.Series(np.isfinite(indicators.to_numpy(dtype=float)).all(axis=1), index=data.index)
        above = ready & data['close'].gt(indicators['ma200'])
        hub = indicators['degree'].ge(params.hub_degree) & indicators['degree_change'].ge(2)
        leaf_recovery = indicators['previous_degree'].eq(1) & indicators['degree'].ge(2)
        entry = above & indicators['close_change'].gt(0) & indicators['atr'].gt(0) & (
            (indicators['ma_slope'].gt(0) & hub)
            | (indicators['ma_slope'].le(0) & leaf_recovery)
        )
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = entry.fillna(False).to_numpy(dtype=bool)
        regime = above.fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Entry fills at this open; freeze ATR known at the preceding close.
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                valid_price = np.isfinite(close[i]) and close[i] > 0
                valid_entry = np.isfinite(fixed_stop) and np.isfinite(opens[entry_bar]) and opens[entry_bar] > 0
                stopped = valid_price and valid_entry and close[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.hold_days
                if not valid_price or not valid_entry or stopped or expired or not regime[i]:
                    entry_bar = -1
                    fixed_stop = np.nan
                else:
                    decisions[i] = 1
            elif candidates[i]:
                decisions[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range('2020-01-01', periods=212, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'ma200': 90.0, 'ma_slope': 1.0, 'atr': 2.0, 'degree': 1.0, 'previous_degree': 1.0, 'degree_change': 0.0, 'close_change': 0.0}, index=index)
    derived.loc[index[205], ['degree', 'degree_change', 'close_change']] = [4.0, 3.0, 1.0]
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    signal = strategy.generate_signals(data, derived, ctx, params).data['signal']
    assert signal.iloc[205:209].tolist() == [0, 1, 1, 0]
    data.loc[index[206], 'close'] = 97.0
    signal = strategy.generate_signals(data, derived, ctx, params).data['signal']
    assert signal.iloc[205:209].tolist() == [0, 1, 0, 0]
    topology_data = data.iloc[:5].copy()
    topology_data['close'] = [4.0, 1.0, 2.0, 3.0, 5.0]
    assert strategy.indicators(topology_data, TopologyParams(graph_window=5))['degree'].iloc[-1] == 2.0
