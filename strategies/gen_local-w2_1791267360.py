from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.55
    capacity_threshold: float = 1.8
    holding_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if not np.isfinite(self.capacity_threshold) or self.capacity_threshold <= 0.0:
            raise ValueError('capacity_threshold must be finite and positive')
        if isinstance(self.holding_bars, bool) or not isinstance(self.holding_bars, int) or not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be an integer from three through five')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791267360'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume > 0.0)
        # ponytail: four-tap truncation limits fractional memory; longer kernels require a larger warmup budget.
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, 4):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        innovation = fractional_close.diff()
        baseline = innovation.rolling(5, min_periods=5).mean().shift(1)
        scale = innovation.rolling(5, min_periods=5).std(ddof=0).shift(1)
        scale = scale.where(scale > 0.0)
        surprise = (innovation - baseline) / scale
        prior_volume = volume.rolling(3, min_periods=3).mean().shift(1)
        relative_volume = volume / prior_volume
        capacity_score = surprise / relative_volume
        capacity_score = capacity_score.replace([np.inf, -np.inf], np.nan)
        reversal = innovation * baseline < 0.0
        qualified = reversal & (capacity_score.abs() >= params.capacity_threshold)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[qualified & (innovation > 0.0)] = 1
        entry.loc[qualified & (innovation < 0.0)] = -1
        return pd.DataFrame({
            'fractional_close': fractional_close,
            'innovation': innovation,
            'baseline': baseline,
            'scale': scale,
            'surprise': surprise,
            'prior_volume': prior_volume,
            'relative_volume': relative_volume,
            'capacity_score': capacity_score,
            'reversal': reversal,
            'qualified': qualified,
            'entry': entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if direction != 0:
                if bar == exit_bar:
                    direction = 0
                else:
                    signals[bar] = direction
                continue
            if entry != 0:
                direction = int(entry)
                exit_bar = bar + params.holding_bars
                signals[bar] = direction
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=11, freq='D')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': [1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result['size'].gt(0.0).all()
    prices = pd.Series(np.arange(20, dtype=float) + 100.0)
    sample = pd.DataFrame({'open': prices, 'high': prices + 1.0, 'low': prices - 1.0, 'close': prices, 'volume': 1000.0})
    derived = strategy.indicators(sample, params)
    assert derived['baseline'].first_valid_index() == strategy.warmup_bars(params) - 1
    assert derived['entry'].iloc[:9].eq(0).all()
