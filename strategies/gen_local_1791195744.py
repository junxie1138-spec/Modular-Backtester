from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    return_bars: int = 5
    envelope_bars: int = 63
    pressure_bars: int = 5
    edge_fraction: float = 0.15
    pressure_threshold: float = 0.15
    hold_bars: int = 18

    def __post_init__(self):
        lengths = (self.return_bars, self.envelope_bars, self.pressure_bars, self.hold_bars)
        if any(type(value) is not int or value < 1 for value in lengths):
            raise ValueError('Lookbacks and holding period must be positive integers')
        if not 0.0 < self.edge_fraction < 0.5:
            raise ValueError('edge_fraction must be between zero and one half')
        if not 0.0 < self.pressure_threshold < 1.0:
            raise ValueError('pressure_threshold must be between zero and one')


def _fixed_holds(entries, hold_bars):
    signals = np.zeros(len(entries), dtype=int)
    position = 0
    entered = -1
    for i, candidate in enumerate(entries):
        if position:
            if i - entered >= hold_bars:
                position = 0
        elif candidate:
            position = int(candidate)
            entered = i
        signals[i] = position
    return signals


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791195744'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.return_bars + params.envelope_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data['close'].where(data['close'] > 0)
        returns = close.pct_change(periods=params.return_bars, fill_method=None)
        history = returns.shift(1).rolling(params.envelope_bars, min_periods=params.envelope_bars)
        floor = history.min()
        ceiling = history.max()
        width = (ceiling - floor).where(ceiling > floor)
        relative_position = (returns - floor) / width

        candle_range = data['high'] - data['low']
        closing_location = ((2.0 * data['close'] - data['high'] - data['low']) / candle_range.where(candle_range > 0)).clip(-1.0, 1.0)
        closing_location = closing_location.mask(candle_range.eq(0), 0.0)
        volume = data['volume'].where(data['volume'] >= 0)
        weighted_pressure = closing_location * volume
        pressure_sum = weighted_pressure.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        volume_sum = volume.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        pressure = pressure_sum / volume_sum.where(volume_sum > 0)

        return pd.DataFrame({
            'returns': returns,
            'return_floor': floor,
            'return_ceiling': ceiling,
            'return_width': width,
            'relative_position': relative_position,
            'candle_range': candle_range,
            'closing_location': closing_location,
            'weighted_pressure': weighted_pressure,
            'pressure_sum': pressure_sum,
            'volume_sum': volume_sum,
            'pressure': pressure,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        ready = indicators[['returns', 'relative_position', 'pressure']].notna().all(axis=1)
        buy = ready & indicators['returns'].lt(0) & indicators['relative_position'].le(params.edge_fraction) & indicators['pressure'].ge(params.pressure_threshold)
        sell = ready & indicators['returns'].gt(0) & indicators['relative_position'].ge(1.0 - params.edge_fraction) & indicators['pressure'].le(-params.pressure_threshold)
        entries = np.select([buy, sell], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_holds(entries, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    assert _fixed_holds([1, -1, -1, -1, -1, 1, 1], 3).tolist() == [1, 1, 1, 0, -1, -1, -1]
    assert _fixed_holds([0, -1, 1, 0, 0], 2).tolist() == [0, -1, -1, 0, 0]
