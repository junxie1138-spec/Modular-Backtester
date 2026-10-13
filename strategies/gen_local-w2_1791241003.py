from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRetentionParams:
    baseline_bars: int = 40
    memory_bars: int = 12
    relaxation_bars: int = 3
    volatility_bars: int = 20
    shock_ratio: float = 1.4
    relaxation_ratio: float = 0.8
    retention_threshold: float = 0.04
    hold_bars: int = 18
    target_volatility: float = 0.12
    minimum_size: float = 0.05
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeRetentionParams]):
    strategy_id = 'gen_local-w2_1791241003'

    @classmethod
    def params_type(cls) -> type[RangeRetentionParams]:
        return RangeRetentionParams

    def warmup_bars(self, params: RangeRetentionParams) -> int:
        return int(max(
            params.baseline_bars + params.memory_bars - 1,
            params.memory_bars + 1,
            params.relaxation_bars,
            params.volatility_bars,
        ))

    def indicators(self, data: pd.DataFrame, params: RangeRetentionParams) -> pd.DataFrame:
        if min(params.baseline_bars, params.memory_bars, params.relaxation_bars,
               params.volatility_bars, params.hold_bars) < 1:
            raise ValueError('Bar counts must be positive.')
        if not (0 < params.minimum_size <= params.maximum_size <= 1):
            raise ValueError('Size bounds must satisfy 0 < minimum <= maximum <= 1.')
        if not (params.target_volatility > 0 and params.retention_threshold > 0
                and params.shock_ratio > 1 and 0 < params.relaxation_ratio < 1):
            raise ValueError('Invalid volatility or entry parameters.')

        out = pd.DataFrame(index=data.index)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['range_fraction'] = (data['high'] - data['low']).clip(lower=0).div(
            out['midpoint'].where(out['midpoint'] > 0)
        )
        out['baseline_range'] = out['range_fraction'].rolling(params.baseline_bars).mean()
        out['relative_range'] = out['range_fraction'].div(
            out['baseline_range'].where(out['baseline_range'] > 0)
        )
        out['remembered_shock'] = out['relative_range'].rolling(params.memory_bars).max()
        out['relaxed_range'] = out['relative_range'].rolling(params.relaxation_bars).mean()
        out['range_budget'] = (data['high'] - data['low']).clip(lower=0).rolling(
            params.memory_bars
        ).sum()
        out['midpoint_displacement'] = out['midpoint'].diff(params.memory_bars)
        out['retention'] = out['midpoint_displacement'].div(
            out['range_budget'].where(out['range_budget'] > 0)
        )
        out['log_range'] = np.log(
            data['high'].where(data['high'] > 0).div(data['low'].where(data['low'] > 0))
        )
        out['range_variance'] = out['log_range'].pow(2).div(4.0 * np.log(2.0))
        out['annual_volatility'] = np.sqrt(
            out['range_variance'].rolling(params.volatility_bars).mean() * 252.0
        )
        out['conviction'] = (
            out['retention'] / (2.0 * params.retention_threshold)
        ).clip(lower=0, upper=1)
        out['entry_size'] = (
            params.target_volatility
            / out['annual_volatility'].where(out['annual_volatility'] > 0)
            * out['conviction']
        ).clip(lower=params.minimum_size, upper=params.maximum_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeRetentionParams,
    ) -> SignalFrame:
        ready = indicators[['remembered_shock', 'relaxed_range', 'retention',
                            'annual_volatility', 'entry_size']].notna().all(axis=1)
        eligible = (
            ready
            & (indicators['remembered_shock'] >= params.shock_ratio)
            & (indicators['relaxed_range'] <= params.relaxation_ratio)
            & (indicators['retention'] >= params.retention_threshold)
            & (indicators['annual_volatility'] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators['entry_size'].fillna(params.minimum_size).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        exit_bar = -1
        frozen_size = params.minimum_size

        for i in range(len(data)):
            if i == exit_bar:
                # The shifted flat target exits exactly hold_bars after the fill.
                continue
            if i < exit_bar:
                signals[i] = 1
                sizes[i] = frozen_size
            elif eligible[i]:
                exit_bar = i + params.hold_bars
                frozen_size = float(entry_sizes[i])
                signals[i] = 1
                sizes[i] = frozen_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RangeRetentionParams()
    index = pd.date_range('2020-01-01', periods=30, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'remembered_shock': 0.0, 'relaxed_range': 0.5,
                        'retention': 0.1, 'annual_volatility': 0.2,
                        'entry_size': 0.4}, index=index)
    ind.iloc[2, ind.columns.get_loc('remembered_shock')] = 2.0
    # A second trigger during the hold must neither resize nor extend it.
    ind.iloc[5, ind.columns.get_loc('remembered_shock')] = 2.0
    ind.iloc[5, ind.columns.get_loc('entry_size')] = 0.9
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[3:3 + params.hold_bars] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert np.all(result.loc[result['signal'] == 1, 'size'].to_numpy() == 0.4)
    assert (result['size'] > 0).all()
    flat_ind = strategy.indicators(data, params)
    assert not strategy.generate_signals(data, flat_ind, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
