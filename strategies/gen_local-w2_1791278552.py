from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTranslationParams:
    atr_window: int = 4
    strength_threshold: float = 0.15

    def __post_init__(self):
        if not 1 <= self.atr_window <= 9:
            raise ValueError('atr_window must be between 1 and 9')
        if not np.isfinite(self.strength_threshold) or self.strength_threshold <= 0:
            raise ValueError('strength_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[RangeTranslationParams]):
    strategy_id = 'gen_local-w2_1791278552'

    @classmethod
    def params_type(cls):
        return RangeTranslationParams

    @staticmethod
    def warmup_bars(params: RangeTranslationParams) -> int:
        return int(params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeTranslationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['high_step'] = data['high'].diff()
        out['low_step'] = data['low'].diff()
        out['range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['range', 'high_gap', 'low_gap']].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        upward = (out['high_step'] > 0) & (out['low_step'] > 0)
        downward = (out['high_step'] < 0) & (out['low_step'] < 0)
        out['translation'] = np.select(
            [upward, downward],
            [out[['high_step', 'low_step']].min(axis=1),
             out[['high_step', 'low_step']].max(axis=1)],
            default=0.0,
        )
        out['strength'] = (out['translation'] / out['atr'].where(out['atr'] > 0)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTranslationParams,
    ) -> SignalFrame:
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        entries = np.select(
            [strength >= params.strength_threshold,
             strength <= -params.strength_threshold],
            [1, -1], default=0,
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RangeTranslationParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame({'strength': [0.3, 0.0, -0.3, -0.3, 0.0, 0.3, 0.3]}, index=index)
    result = GeneratedStrategy.generate_signals(data, evidence, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result.data['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) <= 10
