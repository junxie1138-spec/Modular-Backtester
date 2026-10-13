from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ceiling_window: int = 20
    min_streak: int = 2
    efficiency_window: int = 5
    min_efficiency: float = 0.35
    atr_window: int = 14
    trail_atr: float = 1.5
    max_hold: int = 4

    def __post_init__(self):
        if min(self.ceiling_window, self.min_streak, self.efficiency_window, self.atr_window, self.max_hold) < 1:
            raise ValueError('Windows, streak threshold, and holding period must be positive.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive.')
        if not np.isfinite(self.min_efficiency) or not 0 <= self.min_efficiency <= 1:
            raise ValueError('min_efficiency must lie between zero and one.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791338120'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.ceiling_window + params.min_streak + 1, params.efficiency_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['ceiling'] = data['high'].rolling(params.ceiling_window, min_periods=params.ceiling_window).max().shift(1)
        out['change'] = close.diff()
        out['contained_advance'] = (close.gt(previous) & close.le(out['ceiling']) & out['ceiling'].notna()).fillna(False)
        groups = (~out['contained_advance']).cumsum()
        out['streak'] = out['contained_advance'].astype(int).groupby(groups).cumsum()
        out['prior_streak'] = out['streak'].shift(1).fillna(0)
        out['path_length'] = out['change'].abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        out['displacement'] = close.diff(params.efficiency_window)
        out['efficiency'] = (out['displacement'] / out['path_length'].replace(0, np.nan)).clip(-1, 1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['strength'] = (0.35 + 0.60 * (out['prior_streak'] / (params.min_streak + 3)).clip(0, 1)).fillna(0.35)
        out['entry'] = (
            close.gt(out['ceiling'])
            & out['prior_streak'].ge(params.min_streak)
            & out['efficiency'].ge(params.min_efficiency)
            & out['atr'].gt(0)
            & np.isfinite(out['atr'])
            & np.isfinite(close)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.95, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.35).clip(0.35, 0.95).to_numpy(dtype=float)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        entry_size = 0.95
        for i in range(count):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                if not valid:
                    active = False
                    continue
                # The first in-trade close follows the entry decision by one bar.
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or i - entry_bar >= params.max_hold:
                    active = False
                    continue
            elif entries[i] and valid:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                entry_size = strengths[i]
            if active:
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 102.0, 101.0, 100.5, 105.0, 106.0]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 5.0, 5.0, 1.0, 1.0],
        'entry': [True, False, False, False, False, False],
        'strength': [0.7] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.loc[index[1], 'size'] == 0.7
    assert result['size'].gt(0).all()
    timed = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams(trail_atr=100.0, max_hold=2)).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
