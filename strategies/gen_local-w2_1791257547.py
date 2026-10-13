from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 24
    bridge_threshold: float = 0.35
    excursion_fraction: float = 0.025
    atr_window: int = 14
    breakeven_fraction: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791257547'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.graph_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_window < 4 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError('Lookbacks and holding period must be positive; graph_window must be at least four.')
        if not 0 < params.bridge_threshold <= 1:
            raise ValueError('bridge_threshold must be in (0, 1].')
        if params.excursion_fraction <= 0 or params.breakeven_fraction <= 0 or params.trail_atr <= 0:
            raise ValueError('Excursion, breakeven, and ATR multiplier must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        valid_close = close.where(np.isfinite(close) & close.gt(0))
        log_price = np.log(valid_close)
        out['log_price'] = log_price
        out['ma200'] = valid_close.rolling(200, min_periods=200).mean()
        previous = valid_close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['prior_peak'] = valid_close.shift(1).rolling(params.graph_window, min_periods=params.graph_window).max()
        out['prior_trough'] = valid_close.shift(1).rolling(params.graph_window, min_periods=params.graph_window).min()
        out['prior_drawdown'] = previous / out['prior_peak'] - 1.0
        out['prior_drawup'] = previous / out['prior_trough'] - 1.0
        lower = pd.Series(np.inf, index=data.index)
        upper = pd.Series(-np.inf, index=data.index)
        long_edges = pd.Series(0.0, index=data.index)
        short_edges = pd.Series(0.0, index=data.index)
        first_bridge = max(2, params.graph_window // 2)
        bridge_count = params.graph_window - first_bridge + 1
        # Natural visibility: every intervening vertex must lie strictly below the connecting line.
        # Reflected prices reverse the inequality for bearish visibility.
        for distance in range(1, params.graph_window + 1):
            slope = (log_price - log_price.shift(distance)) / distance
            if distance >= first_bridge:
                long_edges += (slope < lower).astype(float)
                short_edges += (slope > upper).astype(float)
            lower = pd.concat([lower, slope], axis=1).min(axis=1)
            upper = pd.concat([upper, slope], axis=1).max(axis=1)
        graph_ready = log_price.rolling(params.graph_window + 1, min_periods=params.graph_window + 1).count().eq(params.graph_window + 1)
        out['long_bridge_fraction'] = (long_edges / bridge_count).where(graph_ready)
        out['short_bridge_fraction'] = (short_edges / bridge_count).where(graph_ready)
        ready = out['ma200'].notna() & out['atr'].gt(0) & np.isfinite(out['atr']) & graph_ready
        out['long_setup'] = (
            ready & valid_close.gt(out['ma200'])
            & out['prior_drawdown'].le(-params.excursion_fraction)
            & valid_close.gt(previous)
            & out['long_bridge_fraction'].ge(params.bridge_threshold)
            & out['long_bridge_fraction'].shift(1).lt(params.bridge_threshold)
        ).fillna(False)
        out['short_setup'] = (
            ready & valid_close.lt(out['ma200'])
            & out['prior_drawup'].ge(params.excursion_fraction)
            & valid_close.lt(previous)
            & out['short_bridge_fraction'].ge(params.bridge_threshold)
            & out['short_bridge_fraction'].shift(1).lt(params.bridge_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_setup'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_setup'].fillna(False).to_numpy(dtype=bool)
        commands = np.zeros(len(data), dtype=int)
        position = 0
        entered_at = 0
        entry = 0.0
        stop = 0.0
        favorable_peak = 0.0
        armed = False
        for i in range(len(data)):
            # Yesterday's close decision becomes a position at today's open.
            if position == 0 and i > 0 and commands[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    position = int(commands[i - 1])
                    entered_at = i
                    entry = opens[i]
                    favorable_peak = position * entry
                    stop = favorable_peak - params.trail_atr * atr[i - 1]
                    armed = False
            exited = False
            if position != 0:
                if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    position = 0
                    exited = True
                else:
                    mark = position * close[i]
                    favorable_peak = max(favorable_peak, mark)
                    if favorable_peak - position * entry >= entry * params.breakeven_fraction:
                        armed = True
                    if armed:
                        # Signed-price stops only move up; native short stops only move down.
                        stop = max(stop, position * entry, favorable_peak - params.trail_atr * atr[i])
                    # Stops are close-confirmed; the exit fills at the following open.
                    if mark <= stop or i - entered_at + 1 >= params.max_hold:
                        position = 0
                        exited = True
            if position != 0:
                commands[i] = position
            elif not exited:
                if longs[i]:
                    commands[i] = 1
                elif shorts[i]:
                    commands[i] = -1
        df = pd.DataFrame({'signal': commands, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params(breakeven_fraction=0.02, trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=5)
    for direction in (1, -1):
        closes = np.array([100.0, 101.0, 104.0, 101.0, 100.0])
        opens = np.array([100.0, 100.0, 101.0, 104.0, 101.0])
        if direction == -1:
            closes = 200.0 - closes
            opens = 200.0 - opens
        data = pd.DataFrame({'open': opens, 'high': np.maximum(opens, closes) + 1.0, 'low': np.minimum(opens, closes) - 1.0, 'close': closes, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 1.0, 'long_setup': [direction == 1, False, False, False, False], 'short_setup': [direction == -1, False, False, False, False]}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= 200
