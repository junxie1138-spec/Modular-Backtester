from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 30
    shape_step: int = 2
    analog_count: int = 24
    neighbors: int = 3
    percentile_window: int = 126
    entry_quantile: float = 0.75
    holding_bars: int = 8

    def __post_init__(self):
        if self.ma_window < 2 or self.shape_step < 1:
            raise ValueError('Invalid moving-average or shape window')
        if not 1 <= self.neighbors <= self.analog_count:
            raise ValueError('Invalid analog neighbor count')
        if self.percentile_window < 2 or self.holding_bars < 1:
            raise ValueError('Invalid percentile or holding window')
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError('entry_quantile must be between zero and one')


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791261248'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.ma_window + 4 * params.shape_step
                   + params.analog_count * params.holding_bars
                   + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        opening = data['open'].astype(float)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['price_std'].replace(0.0, np.nan))
        out['z'] = out['z'].replace([np.inf, -np.inf], np.nan)
        step = params.shape_step
        out['z_velocity'] = out['z'] - out['z'].shift(step)
        out['z_curvature'] = out['z'] - 2.0 * out['z'].shift(step) + out['z'].shift(2 * step)
        features = ['z', 'z_velocity', 'z_curvature']
        distances = []
        outcomes = []
        for j in range(params.analog_count):
            lag = params.holding_bars + 1 + 2 * step + j * params.holding_bars
            distance = sum((out[name] - out[name].shift(lag)) ** 2 for name in features) / 3.0
            # Historical fills use the next open and the open exactly H bars later.
            outcome = opening.shift(lag - params.holding_bars - 1) / opening.shift(lag - 1).replace(0.0, np.nan) - 1.0
            outcome = outcome.replace([np.inf, -np.inf], np.nan)
            out[f'analog_distance_{j}'] = distance
            out[f'analog_return_{j}'] = outcome
            valid = np.isfinite(distance.to_numpy()) & np.isfinite(outcome.to_numpy())
            distances.append(np.where(valid, distance.to_numpy(), np.inf))
            outcomes.append(outcome.to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argpartition(distance_matrix, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distance_matrix, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcome_matrix, nearest, axis=1)
        usable = np.isfinite(selected_distances).all(axis=1)
        out['forecast'] = np.where(usable, selected_outcomes.mean(axis=1), np.nan)
        out['match_distance'] = np.where(usable, selected_distances.mean(axis=1), np.nan)
        window = params.percentile_window
        out['forecast_threshold'] = out['forecast'].shift(1).rolling(window).quantile(params.entry_quantile)
        out['distance_threshold'] = out['match_distance'].shift(1).rolling(window).quantile(0.35)
        out['displacement_threshold'] = out['z'].shift(1).rolling(window).quantile(0.35)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        required = ['z', 'forecast', 'match_distance', 'forecast_threshold',
                    'distance_threshold', 'displacement_threshold']
        ready = indicators[required].notna().all(axis=1)
        entry = (ready
                 & (indicators['z'] < indicators['displacement_threshold'])
                 & (indicators['match_distance'] <= indicators['distance_threshold'])
                 & (indicators['forecast'] > indicators['forecast_threshold'])
                 & (indicators['forecast'] > 0.0)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entered_at = -1
        for i in range(len(data)):
            if entered_at >= 0:
                if i - entered_at == params.holding_bars:
                    entered_at = -1
                else:
                    signal[i] = 1
            elif entry[i]:
                entered_at = i
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AnalogParams()
    index = pd.date_range('2020-01-01', periods=2 * params.holding_bars + 5)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'z': -2.0, 'forecast': 0.03,
                              'match_distance': 0.1, 'forecast_threshold': 0.01,
                              'distance_threshold': 0.2,
                              'displacement_threshold': -1.0}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:params.holding_bars + 1].eq(1).all()
    assert result['signal'].iloc[params.holding_bars + 1] == 0
    assert result['signal'].iloc[params.holding_bars + 2] == 1
    assert result['size'].gt(0.0).all()
    derived = strategy.indicators(data, params)
    flat = strategy.generate_signals(data, derived, None, params).data
    assert flat['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
