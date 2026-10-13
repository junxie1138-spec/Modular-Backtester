from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapIslandParams:
    lookback: int = 20
    compression: float = 0.8
    reach_threshold: float = 0.2
    hold_bars: int = 4

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError('lookback must be an integer >= 2')
        if not 0.0 < self.compression <= 1.0:
            raise ValueError('compression must be in (0, 1]')
        if not 0.0 < self.reach_threshold < 1.0:
            raise ValueError('reach_threshold must be in (0, 1)')
        if not isinstance(self.hold_bars, int) or not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be an integer from 3 through 5')


def _fixed_positions(entries, hold_bars):
    positions = np.zeros(len(entries), dtype=int)
    side = 0
    exit_at = -1
    for i, entry in enumerate(entries):
        if side and i == exit_at:
            # Expiry takes priority over entry; this bar must be flat.
            side = 0
        elif side == 0 and entry:
            side = int(entry)
            exit_at = i + hold_bars
        positions[i] = side
    return positions


class GeneratedStrategy(BaseStrategy[GapIslandParams]):
    strategy_id = 'gen_local_1791209038'

    @classmethod
    def params_type(cls):
        return GapIslandParams

    def warmup_bars(self, params: GapIslandParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: GapIslandParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['prior_high'] = data['high'].shift(1)
        out['prior_low'] = data['low'].shift(1)
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['baseline_range'] = out['range'].shift(1).rolling(
            params.lookback, min_periods=params.lookback
        ).median()
        out['range_ratio'] = out['range'] / out['baseline_range'].where(out['baseline_range'] > 0)
        # Positive reach means more exploration above the open than below it.
        out['reach'] = (data['high'] + data['low'] - 2.0 * data['open']) / out['range']
        out['valid'] = (
            np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
            & (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            & data['open'].between(data['low'], data['high'])
            & data['close'].between(data['low'], data['high'])
            & out['range'].notna()
        )
        prior_valid = out['valid'].shift(1).fillna(False).astype(bool)
        upward_island = (data['open'] > out['prior_high']) & (data['low'] > out['prior_high'])
        downward_island = (data['open'] < out['prior_low']) & (data['high'] < out['prior_low'])
        out['gap_direction'] = np.select(
            [upward_island & out['valid'] & prior_valid,
             downward_island & out['valid'] & prior_valid],
            [1, -1], default=0
        ).astype(int)
        compact = out['range_ratio'].le(params.compression) & out['valid']
        out['range_direction'] = np.select(
            [compact & out['reach'].ge(params.reach_threshold),
             compact & out['reach'].le(-params.reach_threshold)],
            [1, -1], default=0
        ).astype(int)
        # Both directional primitives must independently agree.
        out['entry'] = out['gap_direction'].where(
            out['gap_direction'].eq(out['range_direction']), 0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapIslandParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_positions(entries, params.hold_bars)
        # Entry at i+1 and exit at i+hold_bars+1: exactly hold_bars sessions.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    entries = np.array([1, -1, 0, 0, -1, -1, 1, 0, 0, 1, 0, 0])
    shifted = pd.Series(_fixed_positions(entries, 4)).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0]
    assert _fixed_positions(np.array([], dtype=int), 4).size == 0
