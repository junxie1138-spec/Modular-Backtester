from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    episode_bars: int = 6
    minimum_depth: float = 0.012
    repayment_fraction: float = 0.60
    hold_bars: int = 2

    def __post_init__(self):
        if self.episode_bars < 2:
            raise ValueError('episode_bars must be at least 2')
        if not 0.0 < self.minimum_depth < 1.0:
            raise ValueError('minimum_depth must be between zero and one')
        if not np.isfinite(self.repayment_fraction) or self.repayment_fraction <= 0.0:
            raise ValueError('repayment_fraction must be finite and positive')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')


def _fixed_bar_positions(candidates, hold_bars):
    positions = np.zeros(len(candidates), dtype=int)
    direction = 0
    exit_bar = -1
    for i, candidate in enumerate(candidates):
        if direction:
            if i == exit_bar:
                direction = 0
                continue
        elif candidate:
            direction = int(candidate)
            exit_bar = i + hold_bars
        positions[i] = direction
    return positions


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791258447'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(200, params.episode_bars + 2, 21))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['return'] = close.pct_change(fill_method=None)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['volume_baseline'] = volume.rolling(20, min_periods=20).mean().shift(1)
        out['weighted_move'] = out['return'] * volume
        out['selling_debt'] = (-out['weighted_move'].clip(upper=0)).rolling(params.episode_bars, min_periods=params.episode_bars).sum().shift(1)
        out['buying_debt'] = out['weighted_move'].clip(lower=0).rolling(params.episode_bars, min_periods=params.episode_bars).sum().shift(1)
        out['prior_peak'] = close.rolling(params.episode_bars, min_periods=params.episode_bars).max().shift(1)
        out['prior_trough'] = close.rolling(params.episode_bars, min_periods=params.episode_bars).min().shift(1)
        out['prior_drawdown'] = 1.0 - close.shift(1) / out['prior_peak']
        out['prior_drawup'] = close.shift(1) / out['prior_trough'] - 1.0
        out['episode_return'] = (close / close.shift(params.episode_bars) - 1.0).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators.notna().all(axis=1)
        confirmed = data['volume'] > indicators['volume_baseline']
        long_entry = (
            ready & confirmed
            & (data['close'] > indicators['ma200'])
            & (indicators['prior_drawdown'] >= params.minimum_depth)
            & (indicators['episode_return'] < 0)
            & (indicators['selling_debt'] > 0)
            & (indicators['weighted_move'] > 0)
            & (indicators['weighted_move'] >= params.repayment_fraction * indicators['selling_debt'])
        )
        short_entry = (
            ready & confirmed
            & (data['close'] < indicators['ma200'])
            & (indicators['prior_drawup'] >= params.minimum_depth)
            & (indicators['episode_return'] > 0)
            & (indicators['buying_debt'] > 0)
            & (indicators['weighted_move'] < 0)
            & (-indicators['weighted_move'] >= params.repayment_fraction * indicators['buying_debt'])
        )
        candidates = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_bar_positions(candidates, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    candidates = np.array([1, -1, -1, -1, 1, 1, 1])
    positions = _fixed_bar_positions(candidates, 2)
    assert positions.tolist() == [1, 1, 0, -1, -1, 0, 1]
    executed = pd.Series(positions).shift(1).fillna(0).astype(int)
    assert executed.tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert _fixed_bar_positions(np.array([1, -1, -1, 1]), 1).tolist() == [1, 0, -1, 0]
