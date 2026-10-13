from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    range_bars: int = 20
    roc_bars: int = 2
    volatility_bars: int = 20
    acceleration_threshold: float = 0.75
    repair_split: float = 0.45
    minimum_drawdown: float = 0.01
    holding_bars: int = 2
    minimum_size: float = 0.25
    maximum_size: float = 0.95

    def __post_init__(self):
        if min(self.range_bars, self.roc_bars, self.volatility_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if self.holding_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars')
        if not np.isfinite(self.acceleration_threshold) or self.acceleration_threshold <= 0:
            raise ValueError('Acceleration threshold must be finite and positive')
        if not 0.05 < self.repair_split < 0.90:
            raise ValueError('Repair split must lie between 0.05 and 0.90')
        if not 0 < self.minimum_drawdown < 1:
            raise ValueError('Minimum drawdown must lie between zero and one')
        if not 0 < self.minimum_size <= self.maximum_size <= 1:
            raise ValueError('Sizes must satisfy 0 < minimum <= maximum <= 1')


def _fixed_hold(candidates, conviction, holding_bars, minimum_size):
    signals = np.zeros(len(candidates), dtype=int)
    sizes = np.full(len(candidates), minimum_size, dtype=float)
    direction = 0
    exit_bar = -1
    entry_size = minimum_size
    for i in range(len(candidates)):
        if direction:
            if i == exit_bar:
                direction = 0
                continue
        elif candidates[i] != 0:
            direction = int(candidates[i])
            entry_size = float(conviction[i])
            exit_bar = i + holding_bars
        signals[i] = direction
        if direction:
            sizes[i] = entry_size
    return signals, sizes


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791239923'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(params.range_bars, params.volatility_bars + 2, params.roc_bars + 2)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        out['return'] = close / close.shift(1) - 1.0
        out['roc'] = close / close.shift(p.roc_bars) - 1.0
        out['acceleration'] = out['roc'].diff()
        out['volatility'] = out['return'].shift(1).rolling(p.volatility_bars, min_periods=p.volatility_bars).std(ddof=0)
        out['peak'] = close.rolling(p.range_bars, min_periods=p.range_bars).max()
        out['trough'] = close.rolling(p.range_bars, min_periods=p.range_bars).min()
        out['drawdown'] = 1.0 - close / out['peak']
        out['repair'] = (close - out['trough']) / (out['peak'] - out['trough']).replace(0.0, np.nan)
        out['acceleration_score'] = out['acceleration'] / out['volatility'].replace(0.0, np.nan)
        out['previous_score'] = out['acceleration_score'].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        recovering = ready & (out['drawdown'] >= p.minimum_drawdown) & (out['roc'] > 0)
        early = (out['repair'] >= 0.05) & (out['repair'] < p.repair_split)
        mature = (out['repair'] >= p.repair_split) & (out['repair'] <= 0.90)
        rising = (out['acceleration_score'] >= p.acceleration_threshold) & (out['previous_score'] < p.acceleration_threshold)
        falling = (out['acceleration_score'] <= -p.acceleration_threshold) & (out['previous_score'] > -p.acceleration_threshold)
        out['candidate'] = np.select([recovering & early & rising, recovering & mature & falling], [1, -1], default=0).astype(int)
        out['conviction'] = (out['acceleration_score'].abs() / (2.0 * p.acceleration_threshold)).clip(0.0, 1.0).fillna(0.0)
        out['entry_size'] = p.minimum_size + (p.maximum_size - p.minimum_size) * out['conviction']
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        signals, sizes = _fixed_hold(
            indicators['candidate'].to_numpy(dtype=int),
            indicators['entry_size'].to_numpy(dtype=float),
            params.holding_bars,
            params.minimum_size,
        )
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    signals, sizes = _fixed_hold(
        np.array([1, -1, -1, -1, 1, 1, 0]),
        np.array([0.60, 0.90, 0.80, 0.70, 0.50, 0.40, 0.30]),
        2,
        0.25,
    )
    filled = pd.Series(signals).shift(1).fillna(0).astype(int)
    filled_sizes = pd.Series(sizes).shift(1).fillna(0.25)
    assert filled.tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(filled_sizes.iloc[[1, 2, 4, 5]], [0.60, 0.60, 0.70, 0.70])
    assert (filled_sizes > 0).all()
