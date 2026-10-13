from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReactionParams:
    baseline_bars: int = 20
    reaction_bars: int = 2
    gap_threshold: float = 0.8
    location_threshold: float = 0.15

    def __post_init__(self):
        if self.baseline_bars < 2 or self.reaction_bars < 1:
            raise ValueError('Invalid lookback lengths')
        if not np.isfinite(self.gap_threshold) or self.gap_threshold <= 0:
            raise ValueError('gap_threshold must be finite and positive')
        if not np.isfinite(self.location_threshold) or not 0 < self.location_threshold < 1:
            raise ValueError('location_threshold must lie between zero and one')


class GeneratedStrategy(BaseStrategy[ReactionParams]):
    strategy_id = 'gen_local-w2_1791202917'

    @classmethod
    def params_type(cls):
        return ReactionParams

    def warmup_bars(self, params: ReactionParams) -> int:
        return params.baseline_bars + params.reaction_bars + 1

    def indicators(self, data: pd.DataFrame, params: ReactionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = (data['open'] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        daily_range = (data['high'] - data['low']).where(lambda x: x > 0)
        out['location'] = ((data['close'] - data['low']) / daily_range).replace([np.inf, -np.inf], np.nan)
        r = params.reaction_bars
        b = params.baseline_bars
        out['recent_gap'] = out['gap'].rolling(r, min_periods=r).mean()
        out['baseline_gap'] = out['gap'].shift(r).rolling(b, min_periods=b).mean()
        out['gap_scale'] = out['gap'].shift(r).rolling(b, min_periods=b).std(ddof=0)
        out['recent_location'] = out['location'].rolling(r, min_periods=r).mean()
        out['baseline_location'] = out['location'].shift(r).rolling(b, min_periods=b).mean()
        out['gap_change'] = out['recent_gap'] - out['baseline_gap']
        out['location_change'] = out['recent_location'] - out['baseline_location']
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ReactionParams) -> SignalFrame:
        gap_agrees = (
            indicators['baseline_gap'].le(0)
            & indicators['recent_gap'].gt(0)
            & indicators['gap_scale'].gt(0)
            & indicators['gap_change'].gt(params.gap_threshold * indicators['gap_scale'])
        )
        location_agrees = (
            indicators['recent_location'].gt(0.5)
            & indicators['location_change'].gt(params.location_threshold)
        )
        # Agreement is the position condition: either reversal exits, with no timed exit.
        df = pd.DataFrame(index=data.index)
        df['signal'] = (gap_agrees & location_agrees).fillna(False).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'baseline_gap': [np.nan, -0.001, -0.001, -0.001, -0.001, -0.001],
        'recent_gap': [np.nan, 0.002, 0.002, 0.002, 0.002, -0.002],
        'gap_scale': [np.nan, 0.001, 0.001, 0.001, 0.001, 0.001],
        'gap_change': [np.nan, 0.003, 0.003, 0.003, 0.003, -0.001],
        'recent_location': [np.nan, 0.8, 0.8, 0.4, 0.8, 0.8],
        'location_change': [np.nan, 0.3, 0.3, -0.1, 0.3, 0.3],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, ReactionParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
