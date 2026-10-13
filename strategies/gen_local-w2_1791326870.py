from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    scale_bars: int = 40
    shape_bars: int = 6
    history_bars: int = 180
    neighbors: int = 7
    max_distance: float = 0.9
    min_hit_rate: float = 0.65
    profit_target: float = 0.012
    hold_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(self.fractional_window, self.scale_bars, self.shape_bars) < 2:
            raise ValueError('Indicator windows must be at least two')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five')
        first_lag = self.hold_bars + self.shape_bars
        candidate_count = max(0, (self.history_bars - first_lag) // self.hold_bars + 1)
        if not 1 <= self.neighbors <= candidate_count:
            raise ValueError('Insufficient history for requested neighbors')
        if not 0.0 < self.min_hit_rate <= 1.0:
            raise ValueError('min_hit_rate must be in (0, 1]')
        if self.profit_target <= 0.0 or self.max_distance <= 0.0:
            raise ValueError('Target and distance must be positive')


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local-w2_1791326870'

    @classmethod
    def params_type(cls):
        return ShapeParams

    @staticmethod
    def warmup_bars(params):
        return int(params.fractional_window + params.scale_bars +
                   params.shape_bars + params.history_bars + 2)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        opening = data['open'].astype(float).where(data['open'] > 0)
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        out['fractional_mean'] = fractional.rolling(params.scale_bars).mean()
        out['fractional_scale'] = fractional.rolling(params.scale_bars).std(ddof=0)
        out['displacement'] = (
            (fractional - out['fractional_mean']) /
            out['fractional_scale'].where(out['fractional_scale'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['recovery'] = out['displacement'].diff()
        shape_columns = []
        for offset in range(params.shape_bars):
            column = f'shape_{offset}'
            out[column] = out['displacement'].shift(offset)
            shape_columns.append(column)
        out['matured_peak_close'] = close.rolling(params.hold_bars).max()
        lags = list(range(params.hold_bars + params.shape_bars,
                          params.history_bars + 1, params.hold_bars))
        distances = np.full((len(data), len(lags)), np.inf)
        hits = np.zeros((len(data), len(lags)), dtype=float)
        shape = out[shape_columns]
        for column, lag in enumerate(lags):
            differences = shape - shape.shift(lag)
            distance = np.sqrt(differences.pow(2).mean(axis=1, skipna=False))
            entry = opening.shift(lag - 1)
            peak = out['matured_peak_close'].shift(lag - params.hold_bars)
            valid = distance.notna() & entry.notna() & peak.notna()
            distances[:, column] = distance.where(valid, np.inf).to_numpy()
            hits[:, column] = (peak >= entry * (1.0 + params.profit_target)).to_numpy(dtype=float)
        selected = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distances, selected, axis=1)
        selected_hits = np.take_along_axis(hits, selected, axis=1)
        sufficient = np.isfinite(selected_distances).all(axis=1)
        out['analog_distance'] = np.where(sufficient, selected_distances.mean(axis=1), np.nan)
        out['analog_hit_rate'] = np.where(sufficient, selected_hits.mean(axis=1), np.nan)
        out['candidate'] = (
            (out['analog_hit_rate'] >= params.min_hit_rate) &
            (out['analog_distance'] <= params.max_distance) &
            (out['displacement'] < -0.25) &
            (out['recovery'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        candidate = indicators['candidate'].fillna(False).astype(bool)
        confirmed = candidate & candidate.shift(1, fill_value=False)
        decisions = np.zeros(len(data), dtype=int)
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for bar in range(len(data)):
            if active:
                # Yesterday's entry decision fills at today's open.
                if bar == entry_bar:
                    entry_price = opening[bar]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_price and closing[bar] >= entry_price * (1.0 + params.profit_target)
                time_stop = bar - entry_bar + 1 >= params.hold_bars
                if not valid_price or target_hit or time_stop:
                    active = False
                else:
                    decisions[bar] = 1
            elif confirmed.iloc[bar]:
                active = True
                entry_bar = bar + 1
                entry_price = np.nan
                decisions[bar] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = ShapeParams()
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'candidate': [True, True] + [False] * 7}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[3], 'close'] = 102.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0]
    indicators['candidate'] = [True, False] + [False] * 7
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert not result['signal'].any()


if __name__ == '__main__':
    _self_check()
