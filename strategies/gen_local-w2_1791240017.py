from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    box_bars: int = 4
    baseline_bars: int = 20
    compression_ratio: float = 0.80
    volume_multiple: float = 1.30
    probe_fraction: float = 0.15
    hold_bars: int = 2

    def __post_init__(self):
        if not 2 <= self.box_bars <= self.baseline_bars:
            raise ValueError('Require 2 <= box_bars <= baseline_bars')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if not 0 < self.compression_ratio <= 1:
            raise ValueError('compression_ratio must be in (0, 1]')
        if not np.isfinite(self.volume_multiple) or self.volume_multiple <= 1:
            raise ValueError('volume_multiple must be finite and greater than 1')
        if not np.isfinite(self.probe_fraction) or self.probe_fraction <= 0:
            raise ValueError('probe_fraction must be finite and positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791240017'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.box_bars, params.baseline_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['bar_range'] = data['high'] - data['low']
        out['prior_high'] = data['high'].shift(1).rolling(params.box_bars).max()
        out['prior_low'] = data['low'].shift(1).rolling(params.box_bars).min()
        out['midpoint'] = (out['prior_high'] + out['prior_low']) / 2.0
        out['short_range'] = out['bar_range'].shift(1).rolling(params.box_bars).mean()
        out['baseline_range'] = out['bar_range'].shift(1).rolling(params.baseline_bars).mean()
        out['prior_volume'] = data['volume'].shift(1).rolling(params.baseline_bars).mean()
        out['volume_ratio'] = data['volume'] / out['prior_volume'].where(out['prior_volume'] > 0)
        out['body'] = data['close'] - data['open']
        out['move'] = data['close'].diff()
        out['upper_probe'] = data['high'] - out['prior_high']
        out['lower_probe'] = out['prior_low'] - data['low']
        out['probe_threshold'] = params.probe_fraction * out['baseline_range']
        out['compressed'] = (
            (out['baseline_range'] > 0)
            & (out['short_range'] > 0)
            & (out['short_range'] <= params.compression_ratio * out['baseline_range'])
        ).fillna(False)
        out['volume_confirmed'] = (
            np.isfinite(out['volume_ratio'])
            & (out['volume_ratio'] >= params.volume_multiple)
        ).fillna(False)
        out['volume_direction'] = np.select(
            [
                out['volume_confirmed'] & (out['body'] > 0) & (out['move'] > 0),
                out['volume_confirmed'] & (out['body'] < 0) & (out['move'] < 0),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        out['inside_box'] = (
            (data['close'] > out['prior_low'])
            & (data['close'] < out['prior_high'])
        ).fillna(False)
        out['rejection_direction'] = np.select(
            [
                out['inside_box']
                & (out['lower_probe'] > out['probe_threshold'])
                & (data['close'] > out['midpoint']),
                out['inside_box']
                & (out['upper_probe'] > out['probe_threshold'])
                & (data['close'] < out['midpoint']),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        candidates = np.where(
            indicators['compressed'].fillna(False)
            & indicators['volume_direction'].eq(indicators['rejection_direction'])
            & indicators['volume_direction'].isin([-1, 1]),
            indicators['volume_direction'],
            0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for i, candidate in enumerate(candidates):
            if remaining:
                remaining -= 1
                if remaining == 0:
                    position = 0
            elif candidate:
                position = int(candidate)
                remaining = params.hold_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    data = pd.DataFrame(
        {'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 100.0},
        index=pd.date_range('2020-01-01', periods=8),
    )
    strategy = GeneratedStrategy()
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1]),
    ):
        params = Params(hold_bars=hold)
        features = strategy.indicators(data, params)
        assert not features['compressed'].any()
        assert not strategy.generate_signals(data, features, None, params).data['signal'].any()
        features['compressed'] = True
        for direction in (1, -1):
            features['volume_direction'] = direction
            features['rejection_direction'] = direction
            result = strategy.generate_signals(data, features, None, params).data
            assert result['signal'].tolist() == [direction * value for value in expected]
            assert result.index.equals(data.index)
            assert (result['size'] > 0).all()
            features['rejection_direction'] = -direction
            assert not strategy.generate_signals(data, features, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
