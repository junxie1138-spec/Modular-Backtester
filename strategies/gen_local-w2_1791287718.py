from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    ma_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.85
    reset_percentile: float = 0.50
    channel_window: int = 5
    atr_window: int = 14
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = 'gen_local-w2_1791287718'

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return int(max(params.ma_window + params.percentile_window + 1,
                       params.channel_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        if min(params.ma_window, params.percentile_window, params.channel_window,
               params.atr_window, params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive')
        if not 0 < params.reset_percentile < params.entry_percentile < 1:
            raise ValueError('Require 0 < reset_percentile < entry_percentile < 1')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['sd'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['sd'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['abs_z'] = out['z'].abs()
        out['entry_threshold'] = out['abs_z'].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        out['reset_threshold'] = out['abs_z'].shift(1).rolling(params.percentile_window).quantile(params.reset_percentile)
        out['channel_high'] = close.shift(1).rolling(params.channel_window).max()
        out['channel_low'] = close.shift(1).rolling(params.channel_window).min()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        columns = ['z', 'entry_threshold', 'reset_threshold',
                   'channel_high', 'channel_low', 'atr']
        values = indicators[columns].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        armed = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i, (price, row) in enumerate(zip(closes, values)):
            if not np.isfinite(price) or price <= 0 or not np.isfinite(row).all() or row[5] <= 0:
                side = 0
                armed = False
                continue
            z, upper, lower, channel_high, channel_low, atr = row
            if side:
                # Signed-price high water mirrors the long trail for shorts.
                high_water = max(high_water, side * price)
                stop = max(stop, high_water - params.trail_atr * atr)
                if side * price <= stop or i - entry_bar >= params.max_hold_bars:
                    side = 0
                    armed = False
                signals[i] = side
                continue
            if abs(z) <= lower:
                armed = True
            if armed and abs(z) > upper:
                if z > 0 and price > channel_high:
                    side = 1
                elif z < 0 and price < channel_low:
                    side = -1
                if side:
                    armed = False
                    entry_bar = i
                    high_water = side * price
                    stop = high_water - params.trail_atr * atr
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'close': [100., 100., 105., 101., 100., 100., 95., 99., 100.]}, index=index)
    features = pd.DataFrame({
        'z': [0., 2., 2., 2., 0., -2., -2., -2., 0.],
        'entry_threshold': 1., 'reset_threshold': 0.5,
        'channel_high': 99., 'channel_low': 101.,
        'atr': [1., 1., 1., 10., 1., 1., 1., 10., 1.],
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, features, None, BreakoutParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert (result['size'] > 0).all()
