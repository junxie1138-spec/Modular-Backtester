from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalShockParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    normalization_window: int = 42
    seasonal_window: int = 252
    min_phase_observations: int = 12
    seasonal_threshold: float = 0.10
    shock_threshold: float = 0.75
    atr_window: int = 14
    atr_multiple: float = 2.5
    holding_bars: int = 8
    minimum_size: float = 0.15
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalShockParams]):
    strategy_id = 'gen_local_1791211100'

    @classmethod
    def params_type(cls) -> type[SeasonalShockParams]:
        return SeasonalShockParams

    @staticmethod
    def warmup_bars(params: SeasonalShockParams) -> int:
        return int(max(
            params.fractional_window + params.normalization_window + params.seasonal_window,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalShockParams) -> pd.DataFrame:
        lengths = (
            params.fractional_window, params.normalization_window,
            params.seasonal_window, params.min_phase_observations,
            params.atr_window, params.holding_bars,
        )
        if any(n < 1 for n in lengths) or params.normalization_window < 2:
            raise ValueError('Lookbacks must be positive; normalization requires at least two observations.')
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must lie strictly between zero and one.')
        if not (params.seasonal_threshold > 0 and params.shock_threshold > 0 and params.atr_multiple > 0):
            raise ValueError('Signal thresholds and ATR multiple must be positive.')
        if not 0 < params.minimum_size <= params.maximum_size <= 1:
            raise ValueError('Position sizes must satisfy 0 < minimum_size <= maximum_size <= 1.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # ponytail: fixed-width fractional filter; increase the window to retain longer memory.
        fractional_close = close.copy()
        for lag in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        out['fractional_close'] = fractional_close
        out['innovation'] = fractional_close.diff()
        out['innovation_scale'] = out['innovation'].rolling(
            params.normalization_window, min_periods=params.normalization_window
        ).std(ddof=0)
        out['shock'] = (
            out['innovation'] / out['innovation_scale'].where(out['innovation_scale'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['calendar_phase'] = ((data.index.day - 1) // 7).astype(int)
        past = out['shock'].shift(1)
        out['historical_mean'] = past.rolling(
            params.seasonal_window, min_periods=params.seasonal_window
        ).mean()
        out['seasonal_edge'] = np.nan
        for phase in range(5):
            belongs = out['calendar_phase'].eq(phase)
            observations = out['shock'].where(belongs).shift(1)
            total = observations.rolling(params.seasonal_window, min_periods=1).sum()
            count = observations.notna().astype(float).rolling(
                params.seasonal_window, min_periods=1
            ).sum()
            mean = total / count.where(count >= params.min_phase_observations)
            out[f'phase_{phase}_mean'] = mean
            out[f'phase_{phase}_count'] = count
            edge = mean - out['historical_mean']
            out.loc[belongs, 'seasonal_edge'] = edge.loc[belongs]

        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        direction = np.sign(out['seasonal_edge'])
        eligible = (
            out['seasonal_edge'].abs().ge(params.seasonal_threshold)
            & (out['shock'] * direction).le(-params.shock_threshold)
            & out['atr'].gt(0)
        )
        out['entry_signal'] = direction.where(eligible, 0).fillna(0).astype(int)
        out['conviction'] = (
            (out['seasonal_edge'].abs() / params.seasonal_threshold)
            * (out['shock'].abs() / params.shock_threshold)
        ).pow(0.5).replace([np.inf, -np.inf], np.nan)
        out['entry_size'] = (
            params.minimum_size + (params.maximum_size - params.minimum_size)
            * (1 - np.exp(-out['conviction'].clip(lower=0)))
        ).clip(params.minimum_size, params.maximum_size).fillna(params.minimum_size)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalShockParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_signal'].to_numpy(dtype=int)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        position = 0
        age = 0
        highest = -np.inf
        lowest = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        held_size = params.minimum_size

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0 and atr > 0
            if position:
                age += 1
                if valid:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    trailing_floor = max(trailing_floor, highest - params.atr_multiple * atr)
                    short_ceiling = min(short_ceiling, lowest + params.atr_multiple * atr)
                exit_now = (
                    not valid
                    or close <= trailing_floor
                    or (position == -1 and close >= short_ceiling)
                    or age >= params.holding_bars
                )
                if exit_now:
                    position = 0
                    continue
            elif valid and entries[i] != 0:
                position = int(entries[i])
                held_size = float(entry_sizes[i])
                age = 0
                # Entry fills next bar; initialize watermarks from that bar's close.
                highest, lowest = -np.inf, np.inf
                trailing_floor, short_ceiling = -np.inf, np.inf
            signals[i] = position
            sizes[i] = held_size if position else params.minimum_size

        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = SeasonalShockParams(atr_multiple=2.0)
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'close': [100., 104., 103., 101., 100., 100.]}, index=index)
    derived = pd.DataFrame({
        'atr': [1., 1., 10., 1., 1., 1.],
        'entry_signal': [1, 0, 0, 0, 0, 0],
        'entry_size': [0.6] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result['size'].iloc[1] == 0.6
    assert (result['size'] > 0).all()
    data['close'] = [100., 100., 104., 104., 104., 104.]
    derived['atr'] = 1.0
    derived['entry_signal'] = [-1, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
