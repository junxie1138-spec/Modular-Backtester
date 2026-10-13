from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 20
    smoothing: int = 7
    threshold: float = 0.15
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if self.lookback < 3 or self.smoothing < 1:
            raise ValueError('lookback must be >= 3 and smoothing >= 1')
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError('threshold must be in [0, 1)')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('sizes must satisfy 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1790971608'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(params.lookback + params.smoothing - 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        barrier = pd.Series(-np.inf, index=data.index)
        signed_weight = pd.Series(0.0, index=data.index)
        total_weight = pd.Series(0.0, index=data.index)

        # An edge exists when both endpoints exceed every intervening close.
        # Each lag is vectorized over all dates; only past observations are used.
        for lag in range(1, params.lookback):
            past = close.shift(lag)
            visible = close.notna() & past.notna() & (
                (lag == 1) | (np.minimum(close, past) > barrier)
            )
            weight = np.sqrt(lag) * visible.astype(float)
            signed_weight += weight * np.sign(close - past).fillna(0.0)
            total_weight += weight
            barrier = np.maximum(barrier, past.fillna(np.inf))

        complete = close.rolling(params.lookback, min_periods=params.lookback).count().eq(params.lookback)
        rank = (signed_weight / total_weight.replace(0.0, np.nan)).where(complete)
        score = rank.rolling(params.smoothing, min_periods=params.smoothing).mean()
        return pd.DataFrame({
            'visible_weight': total_weight,
            'signed_visible_weight': signed_weight,
            'relative_rank': rank,
            'score': score,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        score = indicators['score']
        valid = score.notna() & np.isfinite(score)
        entry = score.gt(params.threshold)
        # Remain invested until the same entry condition flips false.
        # Missing observations preserve the previous decision and exposure.
        state = entry.astype(float).where(valid).ffill().fillna(0.0)
        conviction = ((score - params.threshold) / (1.0 - params.threshold)).clip(0.0, 1.0)
        decision_size = params.min_size + (params.max_size - params.min_size) * conviction
        decision_size = decision_size.where(valid).ffill().fillna(params.min_size)
        df = pd.DataFrame(index=data.index)
        df['signal'] = state.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = decision_size.shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=100)
    close = pd.Series(np.arange(1.0, 101.0), index=index)
    data = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1000.0})
    indicators = GeneratedStrategy.indicators(data, params)
    assert indicators['score'].first_valid_index() == index[GeneratedStrategy.warmup_bars(params) - 1]
    assert indicators['score'].iloc[-1] == 1.0
    prefix = GeneratedStrategy.indicators(data.iloc[:70], params)
    assert prefix.equals(indicators.iloc[:70])
    falling = data.copy()
    falling['close'] = close.iloc[::-1].to_numpy()
    assert GeneratedStrategy.indicators(falling, params)['score'].iloc[-1] == -1.0
    probe = pd.DataFrame({'score': [-0.2, 0.5, 0.7, -0.1, 0.8]}, index=index[:5])
    result = GeneratedStrategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert result['size'].between(params.min_size, params.max_size).all()
    assert result['size'].iloc[3] > result['size'].iloc[2]


if __name__ == '__main__':
    _self_check()
