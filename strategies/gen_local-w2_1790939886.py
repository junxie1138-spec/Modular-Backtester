from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInversionParams:
    lookback: int = 60
    gap_percentile: float = 0.85
    overshoot_min: float = 0.20
    hold_bars: int = 2
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.lookback < 2 or not 0.0 < self.gap_percentile < 1.0:
            raise ValueError('Invalid ranking parameters')
        if self.overshoot_min < 0.0 or self.hold_bars not in (1, 2):
            raise ValueError('Invalid entry or holding parameters')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Invalid position sizes')


class GeneratedStrategy(BaseStrategy[GapInversionParams]):
    strategy_id = 'gen_local-w2_1790939886'

    @classmethod
    def params_type(cls):
        return GapInversionParams

    @staticmethod
    def warmup_bars(params):
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['gap_points'] = data['open'] - out['previous_close']
        out['gap_return'] = out['gap_points'] / out['previous_close'].replace(0.0, np.nan)
        out['gap_direction'] = np.sign(out['gap_points'])
        out['gap_magnitude'] = out['gap_return'].abs()
        out['gap_rank'] = out['gap_magnitude'].rolling(
            params.lookback, min_periods=params.lookback
        ).rank(pct=True)
        out['overshoot'] = (
            -out['gap_direction'] * (data['close'] - out['previous_close'])
            / out['gap_points'].abs().replace(0.0, np.nan)
        ).clip(lower=0.0)
        out['rarity_strength'] = (
            (out['gap_rank'] - params.gap_percentile)
            / (1.0 - params.gap_percentile)
        ).clip(lower=0.0, upper=1.0)
        out['entry_size'] = params.min_size + (
            params.max_size - params.min_size
        ) * (
            0.5 * out['rarity_strength']
            + 0.5 * out['overshoot'].clip(upper=1.0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out[['gap_rank', 'overshoot']].notna().all(axis=1)
        out = out.fillna(0.0)
        out['entry_size'] = out['entry_size'].clip(params.min_size, params.max_size)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        eligible = (
            indicators['ready'].astype(bool)
            & (indicators['gap_rank'] >= params.gap_percentile)
            & (indicators['overshoot'] >= params.overshoot_min)
            & (indicators['overshoot'] > 0.0)
        )
        entries = np.where(eligible, -indicators['gap_direction'], 0).astype(int)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position = 0
        entry_bar = -1
        active_size = 1.0
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar >= params.hold_bars:
                    position = 0
                    active_size = 1.0
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
                active_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = active_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapInversionParams(lookback=3)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [100.0] * 8,
        'high': [101.0] * 8,
        'low': [99.0] * 8,
        'close': [100.0] * 8,
        'volume': [1000.0] * 8,
    }, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived.isna().any().any()
    assert GeneratedStrategy.warmup_bars(params) == 4
    probe = pd.DataFrame({
        'ready': [False, False, True, True, True, False, False, False],
        'gap_rank': [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        'overshoot': [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        'gap_direction': [0, 0, -1, 1, 1, 0, 0, 0],
        'entry_size': [0.25, 0.25, 0.70, 0.90, 0.90, 0.25, 0.25, 0.25],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, probe, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert result['size'].iloc[3:5].eq(0.70).all()
    assert result['size'].gt(0.0).all()
    probe['gap_direction'] = -probe['gap_direction']
    result = GeneratedStrategy.generate_signals(data, probe, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, -1, -1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
