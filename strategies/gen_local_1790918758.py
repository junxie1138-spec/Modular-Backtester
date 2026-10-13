from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 40
    recenter_window: int = 12
    deadband_z: float = 0.8
    recenter_fraction: float = 0.5
    breakout_window: int = 20
    entry_z: float = 1.6
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 18
    full_size_z: float = 3.0
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790918758'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window + params.recenter_window + 2,
                       params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ma_window, params.recenter_window, params.breakout_window,
               params.atr_window, params.max_hold) < 1 or params.ma_window < 2:
            raise ValueError('Lookbacks and holding period must be positive; MA window must exceed one.')
        if not (0.0 < params.minimum_size <= 1.0 and
                0.0 < params.recenter_fraction <= 1.0 and
                0.0 < params.deadband_z < params.entry_z <= params.full_size_z and
                params.trail_atr > 0.0):
            raise ValueError('Invalid deadband, entry threshold, sizing, or trailing distance.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['ma'] = close.rolling(params.ma_window).mean().shift(1)
        out['dispersion'] = close.rolling(params.ma_window).std(ddof=0).shift(1).replace(0.0, np.nan)
        out['z'] = ((close - out['ma']) / out['dispersion']).replace([np.inf, -np.inf], np.nan)
        out['previous_z'] = out['z'].shift(1)
        out['in_deadband'] = out['z'].abs().le(params.deadband_z).astype(float).where(out['z'].notna())
        out['recenter_fraction'] = out['in_deadband'].rolling(params.recenter_window).mean().shift(1)
        out['prior_high'] = close.rolling(params.breakout_window).max().shift(1)
        out['prior_low'] = close.rolling(params.breakout_window).min().shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        centered = out['recenter_fraction'].ge(params.recenter_fraction)
        upward = centered & out['z'].ge(params.entry_z) & out['previous_z'].lt(params.entry_z) & close.gt(out['prior_high'])
        downward = centered & out['z'].le(-params.entry_z) & out['previous_z'].gt(-params.entry_z) & close.lt(out['prior_low'])
        out['entry'] = np.select([upward, downward], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.minimum_size, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        z = indicators['z'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0
        for i in range(count):
            price = close[i]
            volatility = atr[i]
            if not (np.isfinite(price) and np.isfinite(volatility) and
                    np.isfinite(z[i]) and price > 0.0 and volatility > 0.0):
                position = 0
                age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    extreme = max(extreme, price)
                    stop = max(stop, extreme - params.trail_atr * volatility)
                    stopped = price <= stop
                else:
                    # Short-side mirror: lowest close since entry and a downward-ratcheting stop.
                    extreme = min(extreme, price)
                    stop = min(stop, extreme + params.trail_atr * volatility)
                    stopped = price >= stop
                if stopped or age >= params.max_hold:
                    position = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                extreme = price
                stop = price - position * params.trail_atr * volatility
            signals[i] = position
            if position:
                sizes[i] = float(np.clip(position * z[i] / params.full_size_z,
                                         params.minimum_size, 1.0))
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=9)
    close = np.array([100., 104., 103., 101., 100., 96., 97., 99., 100.])
    data = pd.DataFrame({'open': close, 'high': close + 1., 'low': close - 1.,
                         'close': close, 'volume': 1000.}, index=index)
    derived = pd.DataFrame({
        'atr': [1., 1., 10., 10., 1., 1., 10., 10., 1.],
        'z': [2., 3., 1., 1., -2., -3., -1., -1., -1.],
        'entry': [1, 0, 0, 0, -1, 0, 0, 0, 0]
    }, index=index)
    context = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, derived, context, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result['size'].between(params.minimum_size, 1.0).all()
    assert result['size'].iloc[2] > result['size'].iloc[1]
    params.max_hold = 2
    timed = strategy.generate_signals(data, derived, context, params).data
    assert timed['signal'].iloc[3] == 0
