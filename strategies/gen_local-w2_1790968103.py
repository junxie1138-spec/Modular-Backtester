from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    history_bars: int = 252
    neighbors: int = 16
    edge_threshold: float = 0.008
    hold_bars: int = 18

    def __post_init__(self):
        if self.history_bars < 24:
            raise ValueError('history_bars must be at least 24')
        if not 1 <= self.neighbors <= self.history_bars:
            raise ValueError('neighbors must be between 1 and history_bars')
        if not 15 <= self.hold_bars <= 20:
            raise ValueError('hold_bars must be between 15 and 20')
        if not np.isfinite(self.edge_threshold) or self.edge_threshold <= 0:
            raise ValueError('edge_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = 'gen_local-w2_1790968103'

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    @staticmethod
    def warmup_bars(params):
        # Oldest analog lag plus six-bar shape lag and capped eight-bar streak.
        return int(params.history_bars + params.hold_bars + 16)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        opening = data['open'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        out['change'] = close.diff()
        out['direction'] = np.sign(out['change'])
        direction = out['direction'].fillna(0)
        groups = direction.ne(direction.shift()).cumsum()
        count = direction.groupby(groups, sort=False).cumcount().add(1).clip(upper=8)
        out['streak'] = (direction * count).where(out['change'].notna())
        out['streak_lag3'] = out['streak'].shift(3)
        out['streak_lag6'] = out['streak'].shift(6)
        features = out[['streak', 'streak_lag3', 'streak_lag6']]
        current = features.to_numpy(dtype=float)
        n = len(data)
        out['forecast'] = np.nan
        out['analog_distance'] = np.nan
        if n == 0:
            return out

        # ponytail: O(bars * history) memory; batch rows if intraday histories become necessary.
        distances = np.full((n, params.history_bars), np.inf)
        outcomes = np.full((n, params.history_bars), np.nan)
        for j in range(params.history_bars):
            lag = params.hold_bars + 2 + j
            historical = features.shift(lag).to_numpy(dtype=float)
            distance = np.square(current - historical).sum(axis=1)
            # An analog decided at s enters at open[s+1], exits at open[s+hold+1].
            # Even the newest label ends at t-1, so all training labels are mature.
            outcome = (opening.shift(lag - params.hold_bars - 1)
                       / opening.shift(lag - 1) - 1.0).to_numpy(dtype=float)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[:, j] = np.where(valid, distance, np.inf)
            outcomes[:, j] = np.where(valid, outcome, np.nan)

        # Stable ordering prefers the more recent analog when distances tie.
        selected = np.argsort(distances, axis=1, kind='stable')[:, :params.neighbors]
        nearest_distances = np.take_along_axis(distances, selected, axis=1)
        nearest_outcomes = np.take_along_axis(outcomes, selected, axis=1)
        ready = np.isfinite(nearest_distances).all(axis=1)
        ready &= np.arange(n) >= GeneratedStrategy.warmup_bars(params) - 1
        out['forecast'] = np.where(ready, nearest_outcomes.mean(axis=1), np.nan)
        out['analog_distance'] = np.where(ready, nearest_distances.mean(axis=1), np.nan)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        forecast = indicators['forecast'].reindex(data.index)
        candidate = pd.Series(0, index=data.index, dtype=int)
        valid = np.isfinite(forecast)
        candidate.loc[valid & (forecast > params.edge_threshold)] = 1
        candidate.loc[valid & (forecast < -params.edge_threshold)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar == params.hold_bars:
                    position = 0
            elif confirmed[i] != 0:
                position = int(confirmed[i])
                entry_bar = i
            positions[i] = position

        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = StreakAnalogParams()
    index = pd.date_range('2020-01-01', periods=48, freq='B')
    data = pd.DataFrame(index=index)
    forecasts = pd.DataFrame({'forecast': -0.02}, index=index)
    forecasts.iloc[:3, 0] = 0.02
    result = GeneratedStrategy.generate_signals(data, forecasts, None, params).data
    signal = result['signal'].to_numpy()
    assert (signal[:2] == 0).all()
    assert (signal[2:20] == 1).all()
    assert signal[20] == 0
    assert (signal[21:39] == -1).all()
    assert signal[39] == 0
    forecasts.iloc[:, 0] = np.nan
    forecasts.iloc[0, 0] = 0.02
    isolated = GeneratedStrategy.generate_signals(data, forecasts, None, params).data
    assert isolated['signal'].eq(0).all()
