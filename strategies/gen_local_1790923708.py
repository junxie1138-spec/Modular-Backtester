from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    lookback: int = 6
    z_threshold: float = 1.1
    gap_noise_threshold: float = 0.5
    hold_bars: int = 2

    def __post_init__(self):
        if not 3 <= self.lookback <= 9:
            raise ValueError('lookback must be between 3 and 9')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if not np.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError('z_threshold must be finite and positive')
        if not np.isfinite(self.gap_noise_threshold) or self.gap_noise_threshold <= 0:
            raise ValueError('gap_noise_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = 'gen_local_1790923708'

    @classmethod
    def params_type(cls):
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        prior = out['prior_close'].rolling(params.lookback, min_periods=params.lookback)
        out['mean'] = prior.mean()
        out['std'] = prior.std(ddof=0).replace(0.0, np.nan)
        out['daily_range'] = (data['high'] - data['low']).clip(lower=0.0)
        out['range_noise'] = out['daily_range'].shift(1).rolling(
            params.lookback, min_periods=params.lookback
        ).mean().replace(0.0, np.nan)
        out['gap'] = data['open'] - out['prior_close']
        out['open_z'] = (data['open'] - out['mean']) / out['std']
        out['close_z'] = (data['close'] - out['mean']) / out['std']
        out['gap_noise'] = out['gap'].abs() / out['range_noise']
        out['repair'] = (data['open'] - data['close']) / out['gap'].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        z = indicators['close_z']
        qualified = (
            indicators[['open_z', 'close_z', 'gap', 'gap_noise', 'repair']].notna().all(axis=1)
            & (z.abs() >= params.z_threshold)
            & (indicators['open_z'].abs() > z.abs())
            & (indicators['gap'] * z > 0.0)
            & (indicators['gap_noise'] >= params.gap_noise_threshold)
            & indicators['repair'].between(0.15, 0.65)
        )
        entries = np.where(qualified, np.where(z > 0.0, -1, 1), 0)
        positions = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if side != 0 and i == exit_bar:
                side = 0
            elif side == 0 and entry != 0:
                side = int(entry)
                exit_bar = i + params.hold_bars
            positions[i] = side
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame(index=index)
    sides = np.array([1, -1, -1, -1, -1, 1, 1, 1])
    derived = pd.DataFrame({
        'open_z': -3.0 * sides,
        'close_z': -2.0 * sides,
        'gap': -1.0 * sides,
        'gap_noise': 1.0,
        'repair': 0.3,
    }, index=index)
    for hold, expected in (
        (1, [0, 1, 0, -1, 0, -1, 0, 1]),
        (2, [0, 1, 1, 0, -1, -1, 0, 1]),
    ):
        result = GeneratedStrategy.generate_signals(
            None, data, derived, None, GapRepairParams(hold_bars=hold)
        )
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0.0).all()
