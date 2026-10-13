from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    range_window: int = 20
    fast_window: int = 3
    atr_window: int = 20
    pressure_window: int = 5
    expansion_ratio: float = 1.2
    headroom_atr: float = 0.2
    pressure_threshold: float = 0.2
    initial_stop_atr: float = 2.0
    breakeven_gain: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = 'gen_local-w2_1791217423'

    @classmethod
    def params_type(cls) -> type[CapacityParams]:
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(max(params.range_window, params.fast_window, params.atr_window, params.pressure_window) + 1)

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        windows = (params.range_window, params.fast_window, params.atr_window, params.pressure_window)
        if any(w < 1 for w in windows) or params.max_hold < 1:
            raise ValueError('Lookbacks and max_hold must be positive.')
        values = (params.expansion_ratio, params.headroom_atr, params.pressure_threshold, params.initial_stop_atr, params.breakeven_gain, params.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Parameters must be finite.')
        if params.expansion_ratio <= 1 or params.headroom_atr < 0 or not 0 < params.pressure_threshold <= 1:
            raise ValueError('Invalid entry thresholds.')
        if min(params.initial_stop_atr, params.breakeven_gain, params.trail_atr) <= 0:
            raise ValueError('Exit distances must be positive.')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['fast_atr'] = out['true_range'].rolling(params.fast_window, min_periods=params.fast_window).mean()
        out['expansion'] = out['fast_atr'] / out['atr'].where(out['atr'] > 0)
        out['upper'] = data['high'].rolling(params.range_window, min_periods=params.range_window).max().shift(1)
        out['lower'] = data['low'].rolling(params.range_window, min_periods=params.range_window).min().shift(1)
        out['upper_room'] = (out['upper'] - data['close']) / out['atr'].where(out['atr'] > 0)
        out['lower_room'] = (data['close'] - out['lower']) / out['atr'].where(out['atr'] > 0)
        out['candle_range'] = data['high'] - data['low']
        out['candle_pressure'] = ((2 * data['close'] - data['high'] - data['low']) / out['candle_range'].where(out['candle_range'] > 0)).clip(-1, 1)
        out.loc[out['candle_range'].eq(0), 'candle_pressure'] = 0.0
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0)
        out['weighted_pressure'] = out['candle_pressure'] * out['valid_volume']
        out['pressure_sum'] = out['weighted_pressure'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['volume_sum'] = out['valid_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['pressure'] = out['pressure_sum'] / out['volume_sum'].where(out['volume_sum'] > 0)
        required = ['atr', 'expansion', 'upper_room', 'lower_room', 'pressure']
        out['ready'] = np.isfinite(out[required]).all(axis=1) & out['atr'].gt(0) & out['upper'].gt(out['lower'])
        expanding = out['expansion'].ge(params.expansion_ratio)
        long_entry = out['ready'] & expanding & out['upper_room'].le(params.headroom_atr) & out['pressure'].ge(params.pressure_threshold)
        short_entry = out['ready'] & expanding & out['lower_room'].le(params.headroom_atr) & out['pressure'].le(-params.pressure_threshold)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        desired = np.zeros(len(data), dtype=int)
        direction = 0
        pending = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    direction = pending
                    entry_price = opens[i]
                    # Signed prices let both long and short stops ratchet upward.
                    stop = direction * entry_price - params.initial_stop_atr * atrs[i - 1]
                    armed = False
                    age = 0
                pending = 0
            exited = False
            if direction:
                age += 1
                signed_close = direction * closes[i]
                # Close-observed stops produce next-bar exits under the contract.
                if not np.isfinite(closes[i]) or closes[i] <= 0 or signed_close <= stop or age >= params.max_hold:
                    direction = 0
                    exited = True
                else:
                    if direction * (closes[i] / entry_price - 1.0) >= params.breakeven_gain:
                        armed = True
                    if armed:
                        stop = max(stop, direction * entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, signed_close - params.trail_atr * atrs[i])
            if direction:
                desired[i] = direction
            elif not exited and entries[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                pending = int(entries[i])
                desired[i] = pending
        df = pd.DataFrame({'signal': desired, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = object.__new__(GeneratedStrategy)
    params = CapacityParams()
    index = pd.date_range('2020-01-01', periods=5)
    for side in (1, -1):
        prices = np.array([100.0, 100.0, 102.0, 99.5, 99.5])
        if side == -1:
            prices = 200.0 - prices
        data = pd.DataFrame({'open': [100.0] * 5, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': [1000.0] * 5}, index=index)
        derived = pd.DataFrame({'atr': [1.0] * 5, 'entry': [side, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, 0]
        assert result['size'].gt(0).all()
