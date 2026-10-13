from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRegimeParams:
    range_window: int = 12
    overlap_cutoff: float = 0.55
    expansion_ratio: float = 1.2
    pressure_window: int = 3
    pressure_min: float = 0.1
    atr_window: int = 14
    trail_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[RangeRegimeParams]):
    strategy_id = 'gen_local_1791269052'

    @classmethod
    def params_type(cls) -> type[RangeRegimeParams]:
        return RangeRegimeParams

    def warmup_bars(self, params: RangeRegimeParams) -> int:
        return int(max(params.range_window + 2, params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: RangeRegimeParams) -> pd.DataFrame:
        if min(params.range_window, params.pressure_window, params.atr_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not 0 <= params.overlap_cutoff <= 1 or params.expansion_ratio <= 1:
            raise ValueError('Invalid range regime thresholds.')
        if not -1 <= params.pressure_min <= 1 or params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError('Invalid pressure or exit parameters.')
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).clip(lower=0)
        out['previous_range'] = out['range'].shift(1)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['previous_midpoint'] = out['midpoint'].shift(1)
        out['range_baseline'] = out['range'].shift(1).rolling(params.range_window).mean()
        out['previous_baseline'] = out['range_baseline'].shift(1)
        out['intersection'] = (np.minimum(data['high'], data['high'].shift(1)) - np.maximum(data['low'], data['low'].shift(1))).clip(lower=0)
        out['overlap_denominator'] = np.minimum(out['range'], out['previous_range']).replace(0, np.nan)
        out['overlap'] = (out['intersection'] / out['overlap_denominator']).clip(0, 1)
        out['overlap_regime'] = out['overlap'].shift(1).rolling(params.range_window).mean()
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['range'].replace(0, np.nan)).clip(-1, 1)
        out['valid_volume'] = data['volume'].clip(lower=0).where(out['close_location'].notna())
        out['pressure_numerator'] = (out['close_location'] * out['valid_volume']).rolling(params.pressure_window).sum()
        out['pressure_denominator'] = out['valid_volume'].rolling(params.pressure_window).sum()
        out['pressure'] = out['pressure_numerator'] / out['pressure_denominator'].replace(0, np.nan)
        out['previous_close'] = data['close'].shift(1)
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['range', 'high_gap', 'low_gap']].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        restorative = (
            (out['overlap_regime'] >= params.overlap_cutoff)
            & (out['previous_range'] > params.expansion_ratio * out['previous_baseline'])
            & (out['range'] < 0.8 * out['previous_range'])
            & (out['midpoint'] < out['previous_midpoint'])
            & (data['close'] > out['previous_midpoint'])
        )
        continuation = (
            (out['overlap_regime'] < params.overlap_cutoff)
            & (out['range'] > params.expansion_ratio * out['range_baseline'])
            & (out['midpoint'] > out['previous_midpoint'])
            & (data['close'] > out['midpoint'])
        )
        out['range_vote'] = restorative | continuation
        out['pressure_vote'] = out['pressure'] > params.pressure_min
        out['entry'] = (out['range_vote'] & out['pressure_vote'] & out['atr'].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeRegimeParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                if valid:
                    # First update occurs on the entry-fill session's close.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or i - entry_bar >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
            elif valid and entries[i]:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0, 105.0, 104.0, 102.0, 102.0]}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0, 10.0, 10.0, 10.0], 'entry': [True, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, RangeRegimeParams(max_hold=4)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0], 'ATR widening must not loosen the stop.'
    data['close'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    indicators['atr'] = 1.0
    result = strategy.generate_signals(data, indicators, None, RangeRegimeParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0], 'Position must expire after two fill sessions.'
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
