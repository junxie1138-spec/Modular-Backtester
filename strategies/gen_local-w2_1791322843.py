from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    lookback: int = 20
    volume_multiple: float = 1.4
    move_sigma: float = 1.2
    center_tolerance: float = 0.2
    range_multiple: float = 1.3
    edge_fraction: float = 0.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = 'gen_local-w2_1791322843'

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if params.lookback < 2 or params.holding_bars not in (1, 2):
            raise ValueError('lookback must be >= 2; holding_bars must be 1 or 2')
        thresholds = (params.volume_multiple, params.move_sigma,
                      params.center_tolerance, params.range_multiple,
                      params.edge_fraction)
        if not all(np.isfinite(value) and value > 0 for value in thresholds):
            raise ValueError('Thresholds must be finite and positive')
        if params.edge_fraction >= 0.5:
            raise ValueError('edge_fraction must be below 0.5')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['return_squared'] = out['return'].pow(2)
        out['prior_return_rms'] = np.sqrt(
            out['return_squared'].rolling(params.lookback).mean().shift(1)
        )
        out['prior_volume'] = data['volume'].rolling(params.lookback).mean().shift(1)
        out['volume_ratio'] = data['volume'].div(out['prior_volume'].where(out['prior_volume'] > 0))
        out['range'] = data['high'] - data['low']
        out['prior_range'] = out['range'].rolling(params.lookback).mean().shift(1)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['midpoint_change'] = out['midpoint'].diff()
        out['center_drift'] = out['midpoint_change'].abs().div(
            out['prior_range'].where(out['prior_range'] > 0)
        )
        out['range_ratio'] = out['range'].div(out['prior_range'].where(out['prior_range'] > 0))
        out['close_location'] = (close - data['low']).div(out['range'].where(out['range'] > 0))
        out['return_shock'] = out['return'].div(
            out['prior_return_rms'].where(out['prior_return_rms'] > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        volume_confirmed = out['volume_ratio'].ge(params.volume_multiple)
        elastic_range = (
            out['center_drift'].le(params.center_tolerance)
            & out['range_ratio'].ge(params.range_multiple)
        )
        out['long_entry'] = (
            volume_confirmed
            & out['return_shock'].le(-params.move_sigma)
            & elastic_range
            & out['close_location'].le(params.edge_fraction)
        ).fillna(False)
        out['short_entry'] = (
            volume_confirmed
            & out['return_shock'].ge(params.move_sigma)
            & elastic_range
            & out['close_location'].ge(1.0 - params.edge_fraction)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticParams,
    ) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError('holding_bars must be 1 or 2')
        candidates = np.where(
            indicators['long_entry'].fillna(False), 1,
            np.where(indicators['short_entry'].fillna(False), -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        position, exit_at = 0, -1
        for i, candidate in enumerate(candidates):
            if position and i >= exit_at:
                # Force a flat bar even when a new entry condition is present.
                position = 0
            elif not position and candidate:
                position = int(candidate)
                exit_at = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    entries = pd.DataFrame({
        'long_entry': [True, True, True, False, False, False, False, False],
        'short_entry': [False, False, False, True, True, True, False, False],
    }, index=index)
    result = strategy.generate_signals(data, entries, None, ElasticParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert result['size'].gt(0).all()
    result = strategy.generate_signals(
        data, entries, None, ElasticParams(holding_bars=1)
    ).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 0, -1, 0, 0]


if __name__ == '__main__':
    _self_check()
