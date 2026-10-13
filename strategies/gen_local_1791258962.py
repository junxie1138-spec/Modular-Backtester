from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownTideParams:
    anchor_bars: int = 84
    tide_bars: int = 18

    def __post_init__(self):
        if self.anchor_bars < 2 or self.tide_bars < 2:
            raise ValueError('Both lookbacks must be at least two bars.')


class GeneratedStrategy(BaseStrategy[DrawdownTideParams]):
    strategy_id = 'gen_local_1791258962'

    @classmethod
    def params_type(cls) -> type[DrawdownTideParams]:
        return DrawdownTideParams

    def warmup_bars(self, params: DrawdownTideParams) -> int:
        return int(params.anchor_bars + 2 * params.tide_bars)

    def indicators(self, data: pd.DataFrame, params: DrawdownTideParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        result['peak'] = close.rolling(params.anchor_bars, min_periods=params.anchor_bars).max()
        result['depth'] = (1.0 - close / result['peak']).clip(lower=0.0)
        result['recent_area'] = result['depth'].rolling(params.tide_bars, min_periods=params.tide_bars).sum()
        result['previous_area'] = result['recent_area'].shift(params.tide_bars)
        result['imbalance'] = result['previous_area'] - result['recent_area']
        result['phase'] = np.sign(result['imbalance']).fillna(0).astype(int)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownTideParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        phases = indicators['phase'].to_numpy(dtype=int)
        valid = np.isfinite(indicators['imbalance'].to_numpy(dtype=float))
        for i, desired in enumerate(phases):
            if valid[i]:
                if position == 0:
                    position = int(desired)
                elif desired != position:
                    # Reversal exits to cash; a later close may enter the new phase.
                    position = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0] * 6}, index=index)
    indicators = pd.DataFrame({'imbalance': [np.nan, 1.0, 1.0, -1.0, -1.0, 0.0], 'phase': [0, 1, 1, -1, -1, 0]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, DrawdownTideParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1]
    assert result.index.equals(index) and (result['size'] > 0).all()
