from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    lookback: int = 6
    volume_lookback: int = 20
    min_strength: float = 0.25
    min_volume_ratio: float = 1.15
    hold_bars: int = 2

    def __post_init__(self):
        if self.lookback < 2 or self.volume_lookback < 2:
            raise ValueError('Lookbacks must be at least two.')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        if not 0.0 < self.min_strength <= 1.0:
            raise ValueError('min_strength must be in (0, 1].')
        if not np.isfinite(self.min_volume_ratio) or self.min_volume_ratio <= 0.0:
            raise ValueError('min_volume_ratio must be positive and finite.')


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = 'gen_local_1791214721'

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(max(params.lookback, params.volume_lookback) + 1)

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = params.lookback
        out['change'] = data['close'].diff()
        out['volume'] = data['volume'].clip(lower=0).replace([np.inf, -np.inf], np.nan)
        out['up_volume'] = out['volume'].where(out['change'] > 0, 0.0).where(out['change'].notna())
        out['down_volume'] = out['volume'].where(out['change'] < 0, 0.0).where(out['change'].notna())
        out['up_sum'] = out['up_volume'].rolling(length).sum()
        out['down_sum'] = out['down_volume'].rolling(length).sum()
        out['up_square_sum'] = out['up_volume'].pow(2).rolling(length).sum()
        out['down_square_sum'] = out['down_volume'].pow(2).rolling(length).sum()
        out['directional_total'] = out['up_sum'] + out['down_sum']
        out['balance'] = (out['up_sum'] - out['down_sum']) / out['directional_total'].replace(0, np.nan)
        out['net_change'] = out['change'].rolling(length).sum()
        out['path_length'] = out['change'].abs().rolling(length).sum()
        out['efficiency'] = out['net_change'].abs() / out['path_length'].replace(0, np.nan)
        out['up_sessions'] = out['up_sum'].pow(2) / out['up_square_sum'].replace(0, np.nan)
        out['down_sessions'] = out['down_sum'].pow(2) / out['down_square_sum'].replace(0, np.nan)
        out['effective_sessions'] = out['up_sessions'].where(out['balance'] > 0, out['down_sessions'])
        out['strength'] = (out['balance'].abs() * out['efficiency'] * np.sqrt(out['effective_sessions'] / length)).clip(0, 1)
        out['volume_baseline'] = out['volume'].rolling(params.volume_lookback).mean().shift(1)
        out['volume_ratio'] = out['volume'] / out['volume_baseline'].replace(0, np.nan)
        out['direction'] = np.sign(out['balance']).fillna(0).astype(int)
        out['conviction'] = (out['strength'] * (out['volume_ratio'] / 2.0).clip(0, 1)).fillna(0).clip(0, 1)
        out['eligible'] = (
            (out['strength'] >= params.min_strength)
            & (out['effective_sessions'] >= 2.0)
            & (out['volume_ratio'] >= params.min_volume_ratio)
            & (out['direction'] * out['net_change'] > 0)
            & (out['direction'] * out['change'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.2, dtype=float)
        eligible = indicators['eligible'].fillna(False).to_numpy(dtype=bool)
        directions = indicators['direction'].fillna(0).to_numpy(dtype=int)
        convictions = indicators['conviction'].fillna(0).clip(0, 1).to_numpy(dtype=float)
        position = 0
        exit_bar = -1
        entry_size = 0.2
        for i in range(count):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and eligible[i] and directions[i] != 0:
                position = int(directions[i])
                entry_size = float(0.2 + 0.75 * convictions[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
            sizes[i] = entry_size if position != 0 else 0.2
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = ParticipationParams()
    actual = strategy.indicators(data, params)
    assert not actual['eligible'].any()
    supplied = pd.DataFrame({'eligible': [True, True, True, True, True, False, False, False], 'direction': [1, -1, -1, -1, 1, 0, 0, 0], 'conviction': [0.8, 0.1, 0.2, 0.4, 0.9, 0.0, 0.0, 0.0]}, index=index)
    result = strategy.generate_signals(data, supplied, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert np.allclose(result['size'].iloc[[1, 2, 4, 5]], [0.8, 0.8, 0.5, 0.5])
    assert (result['size'] > 0).all()
    params.hold_bars = 1
    supplied['eligible'] = [True, False, False, False, False, False, False, False]
    result = strategy.generate_signals(data, supplied, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
