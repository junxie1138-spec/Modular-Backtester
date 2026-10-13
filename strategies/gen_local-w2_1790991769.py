from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapContainmentParams:
    baseline_bars: int = 5
    expansion_ratio: float = 1.2
    contraction_ratio: float = 0.8


class GeneratedStrategy(BaseStrategy[GapContainmentParams]):
    strategy_id = "gen_local-w2_1790991769"

    @classmethod
    def params_type(cls) -> type[GapContainmentParams]:
        return GapContainmentParams

    def warmup_bars(self, params: GapContainmentParams) -> int:
        return min(8, max(2, int(params.baseline_bars))) + 2

    def indicators(self, data: pd.DataFrame, params: GapContainmentParams) -> pd.DataFrame:
        lookback = min(8, max(2, int(params.baseline_bars)))
        daily_range = (data['high'] - data['low']).where(lambda s: s > 0)
        prior_range = daily_range.shift(1)
        baseline = daily_range.shift(2).rolling(lookback, min_periods=lookback).mean()
        prior_high = data['high'].shift(1)
        result = pd.DataFrame(index=data.index)
        result['daily_range'] = daily_range
        result['prior_range'] = prior_range
        result['baseline_range'] = baseline
        result['prior_high'] = prior_high
        result['expansion'] = prior_range / baseline
        result['contraction'] = daily_range / prior_range
        result['close_location'] = (data['close'] - data['low']) / daily_range
        result['gap_intact'] = (data['open'] > prior_high) & (data['low'] > prior_high)
        result['ready'] = baseline.notna() & prior_range.notna() & daily_range.notna() & prior_high.notna()
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapContainmentParams) -> SignalFrame:
        entry_condition = (
            indicators['ready'].astype(bool)
            & indicators['gap_intact'].astype(bool)
            & (indicators['expansion'] >= params.expansion_ratio)
            & (indicators['contraction'] <= params.contraction_ratio)
            & (indicators['close_location'] >= 0.5)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; its reversal exits to cash.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        assert df['signal'].isin([0, 1]).all() and df['size'].gt(0).all()
        return SignalFrame(data=df, signal_column='signal', size_column='size')
