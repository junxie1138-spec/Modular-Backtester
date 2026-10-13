from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthPhaseParams:
    roc_bars: int = 5
    history_bars: int = 252
    tail_quantile: float = 0.75

    def __post_init__(self):
        if self.roc_bars < 1 or self.history_bars < 60:
            raise ValueError('Lookbacks must be positive; history_bars must be at least 60.')
        if not 0.5 < self.tail_quantile < 1.0:
            raise ValueError('tail_quantile must be between 0.5 and 1.')


class GeneratedStrategy(BaseStrategy[MonthPhaseParams]):
    strategy_id = 'gen_local-w2_1791326418'

    @classmethod
    def params_type(cls):
        return MonthPhaseParams

    def warmup_bars(self, params: MonthPhaseParams) -> int:
        return int(params.history_bars + 2 * params.roc_bars + 1)

    def indicators(self, data: pd.DataFrame, params: MonthPhaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['roc'] = close.pct_change(periods=params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff(params.roc_bars).replace([np.inf, -np.inf], np.nan)
        out['early_month'] = data.index.day <= 15
        out['lower'] = np.nan
        out['upper'] = np.nan
        for early in (True, False):
            mask = out['early_month'].eq(early)
            history = out['acceleration'].where(mask).shift(1).rolling(
                params.history_bars, min_periods=max(10, params.history_bars // 6)
            )
            lower = history.quantile(1.0 - params.tail_quantile)
            upper = history.quantile(params.tail_quantile)
            out.loc[mask, 'lower'] = lower.loc[mask]
            out.loc[mask, 'upper'] = upper.loc[mask]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MonthPhaseParams) -> SignalFrame:
        valid = indicators[['acceleration', 'lower', 'upper']].notna().all(axis=1)
        valid &= indicators['upper'] > indicators['lower']
        extreme = pd.Series(0, index=data.index, dtype=int)
        extreme.loc[valid & (indicators['acceleration'] > indicators['upper'])] = 1
        extreme.loc[valid & (indicators['acceleration'] < indicators['lower'])] = -1
        phase = np.where(indicators['early_month'].to_numpy(), 1, -1)
        desired = extreme.to_numpy() * phase
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(desired):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'acceleration': [2.0, 0.0, -2.0, -2.0, 0.0, 2.0, 0.0],
        'lower': -1.0, 'upper': 1.0, 'early_month': True
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, MonthPhaseParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    indicators['early_month'] = False
    inverted = strategy.generate_signals(data, indicators, None, MonthPhaseParams()).data
    assert inverted['signal'].tolist() == [0, -1, -1, 0, 1, 1, 0]
