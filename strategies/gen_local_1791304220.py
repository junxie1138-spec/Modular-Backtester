from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    memory_bars: int = 32
    breakout_bars: int = 20
    breakout_threshold: float = 0.15
    hold_bars: int = 4
    minimum_size: float = 0.25
    size_slope: float = 0.35
    maximum_size: float = 0.95

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if self.memory_bars < 2 or self.breakout_bars < 2:
            raise ValueError('lookbacks must be at least two')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five')
        if not 0.0 < self.minimum_size <= self.maximum_size <= 1.0:
            raise ValueError('sizes must satisfy 0 < minimum <= maximum <= 1')
        if self.breakout_threshold < 0.0 or self.size_slope <= 0.0:
            raise ValueError('threshold must be nonnegative and sizing slope positive')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791304220'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(params.memory_bars + params.breakout_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: finite memory approximation; increase memory_bars for longer tails.
        for lag in range(1, params.memory_bars):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fractional_close
        out['fractional_ceiling'] = fractional_close.shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max()
        out['innovation_scale'] = fractional_close.diff().shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).std(ddof=0).replace(0.0, np.nan)
        out['raw_ceiling'] = close.shift(1).rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max()
        out['breakout_strength'] = (
            (out['fractional_close'] - out['fractional_ceiling'])
            / out['innovation_scale']
        ).replace([np.inf, -np.inf], np.nan)
        out['entry_candidate'] = (
            out['breakout_strength'].gt(params.breakout_threshold)
            & close.lt(out['raw_ceiling'])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        candidates = indicators['entry_candidate'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['breakout_strength'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        entry_bar = -1
        entry_size = params.minimum_size

        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar == params.hold_bars:
                    entry_bar = -1
                    continue
                signals[bar] = 1
                sizes[bar] = entry_size
            elif candidates[bar] and np.isfinite(strengths[bar]):
                entry_bar = bar
                entry_size = float(np.clip(
                    params.minimum_size + params.size_slope * max(0.0, strengths[bar]),
                    params.minimum_size,
                    params.maximum_size,
                ))
                signals[bar] = 1
                sizes[bar] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({
        'entry_candidate': True,
        'breakout_strength': np.arange(12, dtype=float) + 1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert np.allclose(result['size'].iloc[1:5], 0.60)
    assert (result['size'] > 0.0).all()
    constant = pd.DataFrame({'close': 100.0}, index=pd.date_range('2020-01-01', periods=80))
    assert not GeneratedStrategy.indicators(constant, params)['entry_candidate'].any()
