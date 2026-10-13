from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OpeningShelfParams:
    lookback: int = 6
    min_gap: float = 0.25
    shelf_width: float = 1.2
    holding_bars: int = 4


def _fixed_hold(entries, holding_bars):
    signal = [0] * len(entries)
    exit_at = -1
    for i, enter in enumerate(entries):
        if exit_at >= 0:
            if i < exit_at:
                signal[i] = 1
            else:
                exit_at = -1
        elif enter:
            exit_at = i + holding_bars
            signal[i] = 1
    return signal


class GeneratedStrategy(BaseStrategy[OpeningShelfParams]):
    strategy_id = 'gen_local_1791316669'

    @classmethod
    def params_type(cls):
        return OpeningShelfParams

    def warmup_bars(self, params: OpeningShelfParams) -> int:
        return min(6, max(1, int(params.lookback))) + 4

    def indicators(self, data: pd.DataFrame, params: OpeningShelfParams) -> pd.DataFrame:
        window = min(6, max(1, int(params.lookback)))
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['prior_range'] = out['range'].rolling(window, min_periods=window).mean().shift(1)
        out['gap'] = data['open'] - data['close'].shift(1)
        out['gap_units'] = out['gap'] / out['prior_range']
        out['shelf_high'] = data['open'].rolling(window, min_periods=window).max().shift(1)
        out['shelf_low'] = data['open'].rolling(window, min_periods=window).min().shift(1)
        out['shelf_units'] = (out['shelf_high'] - out['shelf_low']) / out['prior_range']
        out['close_location'] = (data['close'] - data['low']) / out['range']
        out['spike'] = (out['gap_units'].abs() >= 2.0).astype(float).where(out['gap_units'].notna())
        out['recent_spike'] = out['spike'].rolling(3, min_periods=3).max().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['gap_units', 'shelf_units', 'close_location', 'recent_spike']].notna().all(axis=1)
        out['entry'] = (
            ready
            & out['gap_units'].ge(params.min_gap)
            & out['gap_units'].le(1.0)
            & out['shelf_units'].le(params.shelf_width)
            & data['open'].gt(out['shelf_high'])
            & data['close'].ge(data['open'])
            & out['close_location'].ge(0.65)
            & out['recent_spike'].eq(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OpeningShelfParams) -> SignalFrame:
        holding_bars = min(5, max(3, int(params.holding_bars)))
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_hold(entries, holding_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    for hold in (3, 4, 5):
        raw = _fixed_hold([False] + [True] * (hold + 2), hold)
        shifted = [0] + raw[:-1]
        assert shifted[2:2 + hold] == [1] * hold
        assert shifted[2 + hold] == 0
    assert _fixed_hold([], 4) == []
