from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NestedDepthParams:
    local_window: int = 10
    anchor_window: int = 63
    percentile_window: int = 126
    entry_percentile: float = 0.85
    atr_window: int = 14
    breakeven_gain: float = 0.02
    trail_atr: float = 3.0
    max_hold: int = 18

    def __post_init__(self):
        if not 2 <= self.local_window < self.anchor_window:
            raise ValueError('Require 2 <= local_window < anchor_window')
        if self.percentile_window < 2 or self.atr_window < 2 or self.max_hold < 1:
            raise ValueError('Invalid lookback or holding period')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must lie between 0.5 and 1')
        if self.breakeven_gain <= 0 or self.trail_atr <= 0:
            raise ValueError('Exit parameters must be positive')


class GeneratedStrategy(BaseStrategy[NestedDepthParams]):
    strategy_id = 'gen_local_1791289322'

    @classmethod
    def params_type(cls):
        return NestedDepthParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.anchor_window + params.percentile_window + 1,
                       params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['local_peak'] = close.rolling(params.local_window).max()
        out['anchor_peak'] = close.rolling(params.anchor_window).max()
        out['local_trough'] = close.rolling(params.local_window).min()
        out['anchor_trough'] = close.rolling(params.anchor_window).min()
        out['local_depth'] = 1.0 - close / out['local_peak']
        out['anchor_depth'] = 1.0 - close / out['anchor_peak']
        out['local_lift'] = close / out['local_trough'] - 1.0
        out['anchor_lift'] = close / out['anchor_trough'] - 1.0
        out['depth_gap'] = out['anchor_depth'] - out['local_depth']
        out['lift_gap'] = out['anchor_lift'] - out['local_lift']
        history = params.percentile_window
        out['depth_threshold'] = out['depth_gap'].rolling(history).quantile(
            params.entry_percentile).shift(1)
        out['lift_threshold'] = out['lift_gap'].rolling(history).quantile(
            params.entry_percentile).shift(1)
        out['local_depth_median'] = out['local_depth'].rolling(history).median().shift(1)
        out['local_lift_median'] = out['local_lift'].rolling(history).median().shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & out['atr'].gt(0)
        long_entry = (
            ready & out['depth_gap'].gt(out['depth_threshold'])
            & out['depth_gap'].gt(0)
            & out['local_depth'].le(out['local_depth_median'])
        )
        short_entry = (
            ready & out['lift_gap'].gt(out['lift_threshold'])
            & out['lift_gap'].gt(0)
            & out['local_lift'].le(out['local_lift_median'])
        )
        out['entry'] = np.select(
            [long_entry & ~short_entry, short_entry & ~long_entry],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        desired = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = 0.0
        entry_bar = 0
        best = -np.inf
        stop = -np.inf
        armed = False

        for i in range(len(data)):
            # The previous close's order becomes a position at this open.
            if side == 0 and i > 0 and desired[i - 1] != 0:
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    continue
                side = int(desired[i - 1])
                entry_price = opening[i]
                entry_bar = i
                best = side * entry_price
                previous_atr = atr[i - 1]
                stop = best - params.trail_atr * previous_atr
                armed = False

            if side != 0:
                if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                price = side * close[i]
                # Stops are evaluated at the close; liquidation occurs next open.
                if price <= stop or i - entry_bar + 1 >= params.max_hold:
                    side = 0
                    continue
                best = max(best, price)
                if best - side * entry_price >= params.breakeven_gain * entry_price:
                    armed = True
                if armed:
                    # Signed-price stops only rise; short buy-stop prices only fall.
                    stop = max(stop, side * entry_price,
                               best - params.trail_atr * atr[i])
                if price <= stop:
                    side = 0
                    continue
                desired[i] = side
            elif np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                desired[i] = entries[i]

        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = NestedDepthParams(breakeven_gain=0.015, trail_atr=2.5)
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=9, freq='D')
    closes = np.array([100, 102, 104, 100, 100, 98, 96, 100, 100], dtype=float)
    opens = np.array([100, 100, 102, 104, 100, 100, 98, 96, 100], dtype=float)
    data = pd.DataFrame({
        'open': opens, 'high': np.maximum(opens, closes) + 1,
        'low': np.minimum(opens, closes) - 1, 'close': closes,
        'volume': 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0, 'entry': [1, 0, 0, 0, -1, 0, 0, 0, 0]
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    for length in range(1, len(data) + 1):
        prefix = strategy.generate_signals(
            data.iloc[:length], indicators.iloc[:length], None, params).data
        assert prefix['signal'].equals(result['signal'].iloc[:length])
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry'].eq(0).all()
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= params.anchor_window + params.percentile_window + 1


if __name__ == '__main__':
    _self_check()
