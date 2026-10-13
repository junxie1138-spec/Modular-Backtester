from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    lookback: int = 20
    gap_min: float = 0.35
    area_min: float = 0.25
    holding_bars: int = 2
    min_size: float = 0.20
    max_size: float = 0.95

    def __post_init__(self):
        if self.lookback < 2 or self.holding_bars not in (1, 2):
            raise ValueError('Require lookback >= 2 and holding_bars in (1, 2).')
        if not (np.isfinite(self.gap_min) and self.gap_min > 0):
            raise ValueError('gap_min must be finite and positive.')
        if not (np.isfinite(self.area_min) and self.area_min > 0):
            raise ValueError('area_min must be finite and positive.')
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError('Require 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = 'gen_local_1790906928'

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return int(params.lookback + 3)

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr_reference'] = out['true_range'].rolling(params.lookback).mean().shift(1)
        out['volume_reference'] = data['volume'].rolling(params.lookback).mean().shift(1)
        out['daily_range'] = data['high'] - data['low']
        out['close_location'] = (
            (2 * data['close'] - data['high'] - data['low'])
            / out['daily_range'].where(out['daily_range'] > 0)
        ).clip(-1, 1)
        out['relative_volume'] = (
            data['volume'] / out['volume_reference'].where(out['volume_reference'] > 0)
        ).clip(0, 3)
        out['gap'] = (
            (data['open'] - previous_close)
            / out['atr_reference'].where(out['atr_reference'] > 0)
        )
        out['pressure'] = out['close_location'] * out['relative_volume']
        g = out['gap']
        q = out['pressure']
        # Exact antisymmetric level-two signature of three path increments.
        out['area'] = 0.5 * (
            g.shift(2) * q.shift(1) - q.shift(2) * g.shift(1)
            + g.shift(2) * q - q.shift(2) * g
            + g.shift(1) * q - q.shift(1) * g
        )
        out['prior_pressure'] = q.shift(1) + q.shift(2)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['strength'] = (
            (-out['area'] / params.area_min).clip(lower=0)
            * (out['gap'].abs() / params.gap_min)
        ).clip(upper=100).fillna(0)
        out['conviction_size'] = (
            params.min_size + (params.max_size - params.min_size)
            * out['strength'] / (1 + out['strength'])
        ).astype(float)
        eligible = (
            out['area'].notna()
            & (out['area'] <= -params.area_min)
            & (out['gap'].abs() >= params.gap_min)
            & (out['gap'] * out['prior_pressure'] > 0)
        )
        out['candidate'] = (-np.sign(out['gap'].fillna(0))).where(eligible, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPressureParams,
    ) -> SignalFrame:
        candidate = indicators['candidate'].to_numpy(dtype=int)
        conviction_size = indicators['conviction_size'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        exit_at = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if direction:
                if i == exit_at:
                    direction = 0
            elif candidate[i]:
                direction = int(candidate[i])
                exit_at = i + params.holding_bars
                entry_size = float(conviction_size[i])
            signal[i] = direction
            if direction:
                size[i] = entry_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapPressureParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'candidate': [-1, 1, 1, 1, -1, -1, -1, 1],
        'conviction_size': np.linspace(0.3, 0.9, 8),
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, 0, 1, 1, 0, -1]
    assert result['size'].iloc[1] == result['size'].iloc[2] == 0.3
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= params.lookback + 3


if __name__ == '__main__':
    _self_check()
