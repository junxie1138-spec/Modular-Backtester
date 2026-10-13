from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalReboundParams:
    lookback: int = 252
    minimum_edge: float = 0.001

    def __post_init__(self):
        if self.lookback < 30:
            raise ValueError('lookback must be at least 30')
        if not np.isfinite(self.minimum_edge) or self.minimum_edge < 0:
            raise ValueError('minimum_edge must be finite and nonnegative')


class GeneratedStrategy(BaseStrategy[SeasonalReboundParams]):
    strategy_id = 'gen_local_1791325258'

    @classmethod
    def params_type(cls):
        return SeasonalReboundParams

    @staticmethod
    def warmup_bars(params):
        return int(params.lookback + 2)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['previous_return'] = out['return'].shift(1)
        out['weekday'] = data.index.dayofweek
        out['previous_weekday'] = out['weekday'].shift(1)
        out['seasonal_edge'] = np.nan
        out['seasonal_samples'] = 0.0
        ready = out['previous_return'].notna().astype(float).rolling(params.lookback, min_periods=params.lookback).sum().eq(params.lookback)
        out['history_ready'] = ready
        for weekday in range(5):
            # Today's realized return completes yesterday's decision episode.
            episode = out['previous_weekday'].eq(weekday) & out['previous_return'].lt(0) & out['return'].notna()
            out[f'episode_{weekday}'] = episode
            out[f'outcome_{weekday}'] = out['return'].where(episode, 0.0)
            counts = episode.astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
            totals = out[f'outcome_{weekday}'].rolling(params.lookback, min_periods=params.lookback).sum()
            out[f'count_{weekday}'] = counts
            out[f'total_{weekday}'] = totals
            out[f'edge_{weekday}'] = (totals / counts.where(counts >= 6)).where(ready)
            current = out['weekday'].eq(weekday)
            out.loc[current, 'seasonal_edge'] = out.loc[current, f'edge_{weekday}']
            out.loc[current, 'seasonal_samples'] = counts.loc[current]
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        condition = (
            indicators['history_ready'].fillna(False)
            & indicators['return'].lt(0)
            & indicators['seasonal_edge'].gt(params.minimum_edge)
            & indicators['seasonal_samples'].ge(6)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition persists; its reversal is the only exit.
        df['signal'] = condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SeasonalReboundParams(lookback=60, minimum_edge=0.0)
    index = pd.bdate_range('2020-01-01', periods=240)
    returns = np.tile([-0.01, 0.02, -0.01, 0.02, 0.001], 48)
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1000000.0}, index=index)
    strategy = GeneratedStrategy
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, None, params).data
    prefix = data.iloc[:180]
    pd.testing.assert_frame_equal(indicators.iloc[:180], strategy.indicators(prefix, params))
    assert signals['signal'].isin([0, 1]).all()
    assert signals['signal'].sum() > 0
    assert signals['signal'].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert signals.loc[indicators['return'].shift(1).ge(0), 'signal'].eq(0).all()
    assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
