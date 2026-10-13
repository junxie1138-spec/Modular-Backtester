from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStaircaseParams:
    noise_window: int = 5
    streak_bars: int = 3
    min_gap_ratio: float = 0.15
    max_range_ratio: float = 1.20
    hold_bars: int = 18

    def __post_init__(self):
        if not 1 <= self.noise_window <= 9:
            raise ValueError('noise_window must be between 1 and 9')
        if not 2 <= self.streak_bars <= 8:
            raise ValueError('streak_bars must be between 2 and 8')
        if not 15 <= self.hold_bars <= 20:
            raise ValueError('hold_bars must be between 15 and 20')
        if not np.isfinite(self.min_gap_ratio) or self.min_gap_ratio <= 0:
            raise ValueError('min_gap_ratio must be finite and positive')
        if not np.isfinite(self.max_range_ratio) or self.max_range_ratio <= 0:
            raise ValueError('max_range_ratio must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapStaircaseParams]):
    strategy_id = 'gen_local-w2_1791285363'

    @classmethod
    def params_type(cls) -> type[GapStaircaseParams]:
        return GapStaircaseParams

    def warmup_bars(self, params: GapStaircaseParams) -> int:
        return int(params.noise_window + params.streak_bars)

    def indicators(self, data: pd.DataFrame, params: GapStaircaseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['session_range'] = data['high'] - data['low']
        out['noise_range'] = out['session_range'].rolling(
            params.noise_window, min_periods=params.noise_window
        ).mean().shift(1)
        denominator = out['noise_range'].where(out['noise_range'] > 0)
        out['gap'] = data['open'] - out['previous_close']
        out['gap_ratio'] = (out['gap'] / denominator).replace([np.inf, -np.inf], np.nan)
        out['range_ratio'] = (out['session_range'] / denominator).replace([np.inf, -np.inf], np.nan)
        out['quiet'] = out['range_ratio'].le(params.max_range_ratio) & out['range_ratio'].ge(0)
        out['up_gap'] = (
            out['quiet']
            & out['gap_ratio'].ge(params.min_gap_ratio)
            & data['low'].gt(out['previous_close'])
        ).fillna(False)
        out['down_gap'] = (
            out['quiet']
            & out['gap_ratio'].le(-params.min_gap_ratio)
            & data['high'].lt(out['previous_close'])
        ).fillna(False)
        for direction in ('up', 'down'):
            qualified = out[direction + '_gap']
            consecutive = pd.Series(True, index=data.index)
            count = pd.Series(0, index=data.index, dtype=int)
            for lag in range(params.streak_bars):
                consecutive = consecutive & qualified.shift(lag, fill_value=False)
                count = count + consecutive.astype(int)
            out[direction + '_streak'] = count
            reached = count.eq(params.streak_bars)
            out[direction + '_entry'] = reached & ~reached.shift(1, fill_value=False)
        out['entry_direction'] = np.select(
            [out['up_entry'], out['down_entry']], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapStaircaseParams,
    ) -> SignalFrame:
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar, candidate in enumerate(entries):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapStaircaseParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=40, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry_direction': 0}, index=index)
    indicators.iloc[0, 0] = 1
    indicators.iloc[2, 0] = -1
    indicators.iloc[19, 0] = -1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:19].eq(1).all()
    assert result['signal'].iloc[19] == 0
    assert result['signal'].iloc[20:38].eq(-1).all()
    assert result['signal'].iloc[38:].eq(0).all()
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == '__main__':
    _self_check()
