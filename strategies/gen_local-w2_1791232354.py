from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTranslationParams:
    overlap_max: float = 0.60
    width_cap: float = 1.35
    hold_bars: int = 2

    def __post_init__(self):
        if not np.isfinite(self.overlap_max) or not 0.0 < self.overlap_max < 1.0:
            raise ValueError('overlap_max must be between zero and one')
        if not np.isfinite(self.width_cap) or self.width_cap <= 1.0:
            raise ValueError('width_cap must exceed one')
        if type(self.hold_bars) is not int or self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')


class GeneratedStrategy(BaseStrategy[RangeTranslationParams]):
    strategy_id = 'gen_local-w2_1791232354'

    @classmethod
    def params_type(cls) -> type[RangeTranslationParams]:
        return RangeTranslationParams

    @staticmethod
    def warmup_bars(params: RangeTranslationParams) -> int:
        return 200

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeTranslationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['day_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['high_change'] = data['high'].diff()
        out['low_change'] = data['low'].diff()
        out['width_ratio'] = out['day_range'] / out['day_range'].shift(1)
        out['width_max'] = out['width_ratio'].rolling(2, min_periods=2).max()
        out['width_min'] = out['width_ratio'].rolling(2, min_periods=2).min()
        out['intersection'] = (
            np.minimum(data['high'], data['high'].shift(1))
            - np.maximum(data['low'], data['low'].shift(1))
        ).clip(lower=0.0)
        out['overlap'] = out['intersection'] / np.minimum(
            out['day_range'], out['day_range'].shift(1)
        )
        out['overlap_peak'] = out['overlap'].rolling(2, min_periods=2).max()
        out['direction_step'] = np.where(
            (out['high_change'] > 0) & (out['low_change'] > 0), 1,
            np.where((out['high_change'] < 0) & (out['low_change'] < 0), -1, 0)
        )
        out['direction_sum'] = out['direction_step'].rolling(2, min_periods=2).sum()
        out['close_location'] = (data['close'] - data['low']) / out['day_range']
        out = out.replace([np.inf, -np.inf], np.nan)
        out['translation_ready'] = (
            (out['overlap_peak'] <= params.overlap_max)
            & (out['width_max'] <= params.width_cap)
            & (out['width_min'] >= 1.0 / params.width_cap)
        ).fillna(False)
        out['long_setup'] = (
            out['translation_ready']
            & (out['direction_sum'] == 2)
            & (out['close_location'] >= 0.65)
            & (data['close'] > out['ma200'])
        ).fillna(False)
        out['short_setup'] = (
            out['translation_ready']
            & (out['direction_sum'] == -2)
            & (out['close_location'] <= 0.35)
            & (data['close'] < out['ma200'])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTranslationParams,
    ) -> SignalFrame:
        candidates = np.where(
            indicators['long_setup'].fillna(False), 1,
            np.where(indicators['short_setup'].fillna(False), -1, 0)
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    # Preserve the scheduled flat bar before another entry.
                positions[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        # Both entry and scheduled exit execute on the following bar.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    for direction in (1, -1):
        setups = pd.DataFrame(
            {'long_setup': direction == 1, 'short_setup': direction == -1}, index=index
        )
        for hold, expected in (
            (1, [0, 1, 0, 1, 0, 1, 0, 1]),
            (2, [0, 1, 1, 0, 1, 1, 0, 1]),
        ):
            result = GeneratedStrategy.generate_signals(
                None, data, setups, None, RangeTranslationParams(hold_bars=hold)
            )
            assert result.data['signal'].tolist() == [direction * x for x in expected]
            assert (result.data['size'] > 0).all()
