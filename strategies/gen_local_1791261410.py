from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBoundaryParams:
    peak_lookback: int = 63
    memory_bars: int = 20
    confirmation_bars: int = 3
    boundary_depth: float = 0.03
    excursion_multiple: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownBoundaryParams]):
    strategy_id = 'gen_local_1791261410'

    @classmethod
    def params_type(cls) -> type[DrawdownBoundaryParams]:
        return DrawdownBoundaryParams

    def warmup_bars(self, params: DrawdownBoundaryParams) -> int:
        return int(max(200, params.peak_lookback + params.memory_bars,
                       params.peak_lookback + params.confirmation_bars))

    def indicators(self, data: pd.DataFrame, params: DrawdownBoundaryParams) -> pd.DataFrame:
        if min(params.peak_lookback, params.memory_bars,
               params.confirmation_bars, params.hold_bars) < 1:
            raise ValueError('Bar counts must be positive.')
        if not (0.0 < params.boundary_depth < 1.0
                and np.isfinite(params.excursion_multiple)
                and params.excursion_multiple > 1.0
                and params.boundary_depth * params.excursion_multiple < 1.0):
            raise ValueError('Invalid drawdown thresholds.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        out['peak'] = close.rolling(params.peak_lookback,
                                    min_periods=params.peak_lookback).max()
        out['depth'] = 1.0 - close / out['peak']
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['prior_excursion'] = out['depth'].shift(1).rolling(
            params.memory_bars, min_periods=params.memory_bars).max()
        valid_depth = out['depth'].notna()
        deep_side = (out['depth'] >= params.boundary_depth).astype(float).where(valid_depth)
        shallow_side = (out['depth'] < params.boundary_depth).astype(float).where(valid_depth)
        out['prior_deep_count'] = deep_side.shift(1).rolling(
            params.confirmation_bars, min_periods=params.confirmation_bars).sum()
        out['prior_shallow_count'] = shallow_side.shift(1).rolling(
            params.confirmation_bars, min_periods=params.confirmation_bars).sum()
        # A disappearing rolling peak must not masquerade as a price recovery.
        stable_peak = out['peak'].eq(out['peak'].shift(1))
        excursion = out['prior_excursion'] >= params.boundary_depth * params.excursion_multiple
        out['long_entry'] = (
            stable_peak & excursion & (close > out['ma200'])
            & (out['depth'] < params.boundary_depth)
            & (out['prior_deep_count'] == params.confirmation_bars)
        ).fillna(False)
        out['short_entry'] = (
            stable_peak & excursion & (close < out['ma200'])
            & (out['depth'] >= params.boundary_depth)
            & (out['prior_shallow_count'] == params.confirmation_bars)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownBoundaryParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')
        candidates = np.where(indicators['long_entry'].fillna(False), 1,
                              np.where(indicators['short_entry'].fillna(False), -1, 0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'long_entry': False, 'short_entry': False}, index=index)
    entries.loc[index[0], 'long_entry'] = True
    entries.loc[index[[3, 10]], 'short_entry'] = True
    strategy = GeneratedStrategy()
    params = DrawdownBoundaryParams()
    result = strategy.generate_signals(data, entries, None, params).data
    expected = np.zeros(24, dtype=int)
    expected[1:9] = 1
    expected[11:19] = -1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert (result['size'] > 0.0).all()
    warm = strategy.indicators(data, params)
    assert not warm['long_entry'].any() and not warm['short_entry'].any()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
