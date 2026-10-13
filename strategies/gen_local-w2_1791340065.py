from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverflowParams:
    capacity_window: int = 20
    capacity_percentile: float = 0.70
    ceiling_window: int = 2
    rank_window: int = 60
    entry_percentile: float = 0.85

    def __post_init__(self):
        if min(self.capacity_window, self.ceiling_window, self.rank_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not (0.0 < self.capacity_percentile < 1.0 and 0.0 < self.entry_percentile < 1.0):
            raise ValueError('Percentiles must lie strictly between zero and one.')


class GeneratedStrategy(BaseStrategy[OverflowParams]):
    strategy_id = 'gen_local-w2_1791340065'

    @classmethod
    def params_type(cls) -> type[OverflowParams]:
        return OverflowParams

    def warmup_bars(self, params: OverflowParams) -> int:
        return int(max(params.capacity_window, params.ceiling_window) + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: OverflowParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['volume_capacity'] = data['volume'].rolling(
            params.capacity_window, min_periods=params.capacity_window
        ).quantile(params.capacity_percentile).shift(1)
        out['range_ceiling'] = data['high'].rolling(
            params.ceiling_window, min_periods=params.ceiling_window
        ).max().shift(1)
        out['bar_range'] = (data['high'] - data['low']).where(
            data['high'] > data['low']
        )
        out['volume_overflow'] = (
            data['volume'] / out['volume_capacity'].where(out['volume_capacity'] > 0) - 1.0
        ).clip(lower=0.0)
        out['breakout_penetration'] = (
            (data['close'] - out['range_ceiling']) / out['bar_range']
        ).clip(lower=0.0)
        out['overflow_pressure'] = out['volume_overflow'] * out['breakout_penetration']
        out = out.replace([np.inf, -np.inf], np.nan)
        out['pressure_threshold'] = out['overflow_pressure'].rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.entry_percentile).shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OverflowParams,
    ) -> SignalFrame:
        pressure = indicators['overflow_pressure']
        threshold = indicators['pressure_threshold']
        entry_condition = pressure.notna() & threshold.notna() & pressure.gt(0.0) & pressure.gt(threshold)
        df = pd.DataFrame(index=data.index)
        # A false entry condition closes the position; no independent timed exit.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'overflow_pressure': [np.nan, 2.0, 3.0, 0.0, 4.0],
        'pressure_threshold': [1.0] * 5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, OverflowParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert result['size'].gt(0.0).all()
