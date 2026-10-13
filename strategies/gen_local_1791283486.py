from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    history: int = 126
    regime_window: int = 20
    streak_percentile: float = 0.85
    atr_window: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.history, self.regime_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.0 < self.streak_percentile < 1.0:
            raise ValueError('streak_percentile must be between zero and one')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0 or self.hold_bars < 1:
            raise ValueError('Trailing distance and holding period must be positive')


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = 'gen_local_1791283486'

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(params.history + params.regime_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['delta'] = close.diff()
        out['direction'] = np.sign(out['delta']).fillna(0).astype(int)
        out['streak_group'] = out['direction'].ne(out['direction'].shift()).cumsum()
        out['streak'] = out['direction'].groupby(out['streak_group']).cumcount().add(1).where(out['direction'].ne(0), 0).astype(float)
        out.loc[out['delta'].isna(), 'streak'] = np.nan
        out['streak_threshold'] = out['streak'].shift(1).rolling(params.history, min_periods=params.history).quantile(params.streak_percentile)
        out['path_length'] = out['delta'].abs().rolling(params.regime_window, min_periods=params.regime_window).sum()
        out['displacement'] = close.diff(params.regime_window).abs()
        out['efficiency'] = out['displacement'].div(out['path_length'].replace(0.0, np.nan))
        out['regime_threshold'] = out['efficiency'].shift(1).rolling(params.history, min_periods=params.history).quantile(0.5)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out.loc[previous_close.isna(), 'true_range'] = np.nan
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['valid'] = np.isfinite(out[['streak', 'streak_threshold', 'efficiency', 'regime_threshold', 'atr']]).all(axis=1) & np.isfinite(close) & out['atr'].gt(0)
        out['extreme'] = out['valid'] & out['streak'].gt(out['streak_threshold'])
        out['episode_start'] = out['extreme'] & (~out['extreme'].shift(1, fill_value=False) | out['direction'].ne(out['direction'].shift(1)))
        out['continuation_regime'] = out['efficiency'].ge(out['regime_threshold'])
        out['entry'] = (out['direction'] * np.where(out['continuation_regime'], 1, -1)).where(out['episode_start'], 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].to_numpy(dtype=int)
        valid = indicators['valid'].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = high_stop = short_stop = np.nan
        for i in range(len(data)):
            if not valid[i] or not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                age = 0
            elif position:
                age += 1
                high_water = max(high_water, close[i])
                high_stop = max(high_stop, high_water - params.trail_atr * atr[i])
                if position == -1:
                    low_water = min(low_water, close[i])
                    short_stop = min(short_stop, low_water + params.trail_atr * atr[i])
                # Apply the required highest-close stop to every trade.
                # Shorts also have a mirrored protective stop against rising prices.
                if close[i] <= high_stop or (position == -1 and close[i] >= short_stop) or age >= params.hold_bars:
                    position = 0
            elif entry[i]:
                position = int(entry[i])
                age = 0
                high_water = low_water = close[i]
                high_stop = close[i] - params.trail_atr * atr[i]
                short_stop = close[i] + params.trail_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100, 100, 103, 104, 103, 100, 100, 100]}, index=index)
    ind = pd.DataFrame({'atr': 1.0, 'entry': [0, 1, 0, 0, 0, 0, 0, 0], 'valid': True}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    data['close'] = [100, 100, 103, 104, 103, 102, 102, 102]
    ind['atr'] = [1, 1, 1, 1, 10, 10, 10, 10]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    data['close'] = [100, 100, 99, 101, 101, 101, 101, 101]
    ind['atr'] = 1.0
    ind['entry'] = [0, -1, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, -1, -1, 0, 0, 0, 0]
    data['close'] = 100.0
    ind['entry'] = [0, 1, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, StreakParams(hold_bars=3)).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data['close'] = [100, 101, 102, 101, 100, 99, 99, 100]
    data['open'] = data['close']
    data['high'] = data['close'] + 1
    data['low'] = data['close'] - 1
    data['volume'] = 1000.0
    derived = strategy.indicators(data, params)
    assert derived['streak'].iloc[1:].tolist() == [1, 2, 1, 2, 3, 0, 1]
    assert not derived['valid'].any()


if __name__ == '__main__':
    _self_check()
