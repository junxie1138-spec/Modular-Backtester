from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    lookback: int = 60
    volume_window: int = 20
    volume_ratio: float = 1.1
    persistence_edge: float = 0.15
    hold_bars: int = 4

    def __post_init__(self):
        if self.lookback < 12 or self.volume_window < 2:
            raise ValueError('Lookbacks are too short.')
        if not np.isfinite(self.volume_ratio) or self.volume_ratio <= 0:
            raise ValueError('volume_ratio must be positive and finite.')
        if not np.isfinite(self.persistence_edge) or not 0 <= self.persistence_edge <= 2:
            raise ValueError('persistence_edge must be between zero and two.')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five.')


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = 'gen_local_1791332321'

    @classmethod
    def params_type(cls):
        return PropagationParams

    @staticmethod
    def warmup_bars(params: PropagationParams) -> int:
        return int(params.volume_window + params.lookback + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: PropagationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['direction'] = np.sign(out['return'])
        out['volume_baseline'] = data['volume'].rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out['relative_volume'] = data['volume'].div(out['volume_baseline'].where(out['volume_baseline'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['high_volume'] = out['relative_volume'].ge(params.volume_ratio) & out['relative_volume'].notna()
        out['lag_direction'] = out['direction'].shift(1)
        out['lag_relative_volume'] = out['relative_volume'].shift(1)
        out['agreement'] = out['direction'] * out['lag_direction']
        out['valid_transition'] = out['agreement'].notna() & out['direction'].ne(0) & out['lag_direction'].ne(0) & out['lag_relative_volume'].notna()
        out['shock_transition'] = out['valid_transition'] & out['lag_relative_volume'].ge(params.volume_ratio)
        out['quiet_transition'] = out['valid_transition'] & out['lag_relative_volume'].lt(params.volume_ratio)
        for regime in ('shock', 'quiet'):
            mask = out[regime + '_transition']
            out[regime + '_count'] = mask.astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
            out[regime + '_agreement_sum'] = out['agreement'].where(mask, 0.0).rolling(params.lookback, min_periods=params.lookback).sum()
            out[regime + '_persistence'] = out[regime + '_agreement_sum'].div(out[regime + '_count'].where(out[regime + '_count'] >= 6))
        out['persistence_difference'] = out['shock_persistence'] - out['quiet_persistence']
        out['confirmed_move'] = out['high_volume'] & out['high_volume'].shift(1, fill_value=False) & out['direction'].eq(out['lag_direction']) & out['direction'].ne(0) & out['direction'].notna()
        out['entry_ready'] = out['confirmed_move'] & out['shock_persistence'].gt(0) & out['persistence_difference'].ge(params.persistence_edge)
        out['entry_direction'] = out['direction'].where(out['entry_ready'], 0).fillna(0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PropagationParams) -> SignalFrame:
        candidates = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(candidates):
            if position:
                if bar == exit_bar:
                    position = 0
                    continue
                signals[bar] = position
            elif candidate:
                position = int(candidate)
                exit_bar = bar + params.hold_bars
                signals[bar] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = PropagationParams()
    index = pd.date_range('2020-01-01', periods=16)
    data = pd.DataFrame(1.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    candidates = pd.DataFrame({'entry_direction': [0, 0, 1, -1, 0, 0, 1, 0, -1, 1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, candidates, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0]
    assert result['size'].gt(0).all()
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry_direction'].eq(0).all()
