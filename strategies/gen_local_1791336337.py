from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_bars: int = 40
    roc_bars: int = 5
    atr_bars: int = 14
    position_floor: float = 0.55
    acceleration_floor: float = 0.12
    breakeven_gain: float = 0.025
    atr_multiple: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791336337'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.range_bars, 2 * params.roc_bars + 1, params.atr_bars + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.range_bars, params.roc_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.5 < params.position_floor < 0.9:
            raise ValueError('position_floor must lie between 0.5 and 0.9.')
        if not all(np.isfinite(v) and v > 0 for v in (params.acceleration_floor, params.breakeven_gain, params.atr_multiple)):
            raise ValueError('Acceleration, breakeven and ATR parameters must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['range_high'] = data['high'].rolling(params.range_bars, min_periods=params.range_bars).max()
        out['range_low'] = data['low'].rolling(params.range_bars, min_periods=params.range_bars).min()
        out['range_width'] = out['range_high'] - out['range_low']
        out['position'] = (close - out['range_low']) / out['range_width'].where(out['range_width'] > 0)
        out['roc'] = close / close.shift(params.roc_bars) - 1.0
        out['lagged_roc'] = out['roc'].shift(params.roc_bars)
        out['atr_fraction'] = out['atr'] / close
        out['acceleration'] = (out['roc'] - out['lagged_roc']) / out['atr_fraction'].where(out['atr_fraction'] > 0)
        out['valid'] = np.isfinite(out[['position', 'roc', 'acceleration', 'atr']]).all(axis=1) & (out['atr'] > 0)
        out['long_setup'] = out['valid'] & out['position'].between(params.position_floor, 0.9) & (out['roc'] < 0) & (out['acceleration'] >= params.acceleration_floor)
        out['short_setup'] = out['valid'] & out['position'].between(0.1, 1.0 - params.position_floor) & (out['roc'] > 0) & (out['acceleration'] <= -params.acceleration_floor)
        out['long_confirmed'] = out['long_setup'] & out['long_setup'].shift(1, fill_value=False)
        out['short_confirmed'] = out['short_setup'] & out['short_setup'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_confirmed'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_confirmed'].fillna(False).to_numpy(dtype=bool)
        side = pending = age = 0
        entry = peak = stop = 0.0
        armed = False
        for i in range(len(data)):
            if pending:
                # Yesterday's decision fills at today's open.
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    side = pending
                    entry = opens[i]
                    peak = side * entry
                    stop = peak - params.atr_multiple * atrs[i - 1]
                    age = 0
                    armed = False
                pending = 0
            exited = False
            if side:
                age += 1
                if np.isfinite(closes[i]) and closes[i] > 0:
                    favorable_price = side * closes[i]
                    peak = max(peak, favorable_price)
                    if favorable_price - side * entry >= params.breakeven_gain * entry:
                        armed = True
                    if armed:
                        # The stop only increases in favorable-price coordinates;
                        # this lowers the corresponding raw-price stop for shorts.
                        stop = max(stop, side * entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.atr_multiple * atrs[i])
                    # Close-observed stops become flat at the following open.
                    exited = favorable_price <= stop
                exited = exited or age >= params.max_hold_bars
                if exited:
                    side = 0
            if side == 0 and not exited and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = 1 if longs[i] else (-1 if shorts[i] else 0)
            signals[i] = side if side else pending
        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=8)
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0, 0, 0, 3, 4, 2, 1, 0], dtype=float)
        data = pd.DataFrame({'open': 100.0, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index)
        confirmed = np.array([False, True, False, False, False, False, False, False])
        indicators = pd.DataFrame({'atr': 1.0, 'long_confirmed': confirmed if direction == 1 else False, 'short_confirmed': confirmed if direction == -1 else False}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, direction, direction, 0]
        assert (result['size'] > 0).all()
