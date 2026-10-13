from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    atr_fast: int = 3
    atr_slow: int = 20
    history: int = 126
    neighbors: int = 7
    minimum_edge_bps: float = 15.0
    pressure_threshold: float = 0.58


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791316781'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        if not 1 <= params.atr_fast <= params.atr_slow:
            raise ValueError('ATR windows must satisfy 1 <= fast <= slow')
        if not 1 <= params.neighbors <= params.history:
            raise ValueError('Neighbors must be between 1 and history')
        if not np.isfinite(params.minimum_edge_bps):
            raise ValueError('Minimum edge must be finite')
        if not 0.0 <= params.pressure_threshold <= 1.0:
            raise ValueError('Pressure threshold must be between zero and one')
        return int(params.atr_slow + params.history + 9)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr_fast'] = out['true_range'].rolling(params.atr_fast, min_periods=params.atr_fast).mean()
        out['atr_slow'] = out['true_range'].rolling(params.atr_slow, min_periods=params.atr_slow).mean()
        ratio = out['atr_fast'].div(out['atr_slow'].where(out['atr_slow'] > 0))
        out['vol_level'] = np.log(ratio.where(ratio > 0))
        out['vol_change_2'] = out['vol_level'].diff(2)
        out['vol_change_4'] = out['vol_level'].diff(4)
        out['realized_return_4'] = close.div(close.shift(4)).sub(1)
        width = data['high'] - data['low']
        out['close_location'] = ((close - data['low']) / width.where(width > 0)).clip(0, 1)
        out['closing_pressure'] = out['close_location'].rolling(4, min_periods=4).mean()
        features = out[['vol_level', 'vol_change_2', 'vol_change_4']]
        distances = []
        outcomes = []
        # A candidate at lag 5 or older has its four-day outcome known before today.
        for lag in range(5, params.history + 5):
            delta = features - features.shift(lag)
            distance = delta.pow(2).sum(axis=1, min_count=3)
            outcome = out['realized_return_4'].shift(lag - 4)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(distance.where(valid, np.inf).to_numpy())
            outcomes.append(outcome.where(valid, 0.0).to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        selected = np.argpartition(distance_matrix, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distance = np.take_along_axis(distance_matrix, selected, axis=1)
        selected_outcome = np.take_along_axis(outcome_matrix, selected, axis=1)
        ready = np.isfinite(selected_distance).all(axis=1)
        out['analog_mean_distance'] = np.where(ready, selected_distance.mean(axis=1), np.nan)
        out['analog_expected_return'] = np.where(ready, selected_outcome.mean(axis=1), np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        volatility_agrees = indicators['analog_expected_return'].gt(params.minimum_edge_bps / 10000.0)
        pressure_agrees = indicators['closing_pressure'].gt(params.pressure_threshold)
        df = pd.DataFrame(index=data.index)
        # Hold while both entry primitives agree; exit only when that conjunction flips false.
        df['signal'] = (volatility_agrees & pressure_agrees).fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AnalogParams()
    index = pd.date_range('2020-01-01', periods=260, freq='B')
    t = np.arange(len(index), dtype=float)
    close = 100.0 + 0.04 * t + np.sin(t / 3.0)
    width = 1.0 + 0.3 * np.sin(t / 7.0)
    data = pd.DataFrame({'open': close - 0.1, 'high': close + width, 'low': close - width, 'close': close, 'volume': 1000000.0}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:220], params)
    assert np.allclose(full.iloc[:220].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert np.isfinite(full['analog_expected_return'].iloc[-1])
    probe = pd.DataFrame({'analog_expected_return': [0.01, 0.01, -0.01, 0.01, 0.01], 'closing_pressure': [0.8, 0.8, 0.8, 0.8, 0.2]}, index=index[:5])
    result = GeneratedStrategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
