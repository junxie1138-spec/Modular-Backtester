from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePatternParams:
    entropy_window: int = 8
    pressure_window: int = 8
    range_window: int = 20
    compression_limit: float = 0.90
    entropy_limit: float = 0.82
    pressure_threshold: float = 0.10
    conviction_scale: float = 0.08

    def __post_init__(self):
        if min(self.entropy_window, self.pressure_window, self.range_window) < 3:
            raise ValueError('Lookbacks must be at least three bars.')
        values = [self.compression_limit, self.entropy_limit,
                  self.pressure_threshold, self.conviction_scale]
        if not np.isfinite(values).all():
            raise ValueError('Thresholds must be finite.')
        if not (0 < self.compression_limit <= 1 and 0 < self.entropy_limit <= 1):
            raise ValueError('Compression and entropy limits must be in (0, 1].')
        if not (0 < self.pressure_threshold < 1 and self.conviction_scale > 0):
            raise ValueError('Invalid pressure threshold or conviction scale.')


class GeneratedStrategy(BaseStrategy[RangePatternParams]):
    strategy_id = 'gen_local-w2_1791212693'

    @classmethod
    def params_type(cls):
        return RangePatternParams

    def warmup_bars(self, params: RangePatternParams) -> int:
        return int(max(params.entropy_window + 2, params.pressure_window,
                       params.range_window, 5))

    def indicators(self, data: pd.DataFrame, params: RangePatternParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        width = (data['high'] - data['low']).where(lambda x: x > 0)
        out['width'] = width
        out['normalized_range'] = width / data['close'].where(data['close'] > 0)
        out['range_fast'] = out['normalized_range'].rolling(5, min_periods=5).mean()
        out['range_base'] = out['normalized_range'].rolling(
            params.range_window, min_periods=params.range_window).mean()
        out['compression'] = out['range_fast'] / out['range_base'].where(
            out['range_base'] > 0)

        out['range_lag2'] = out['normalized_range'].shift(2)
        out['range_lag1'] = out['normalized_range'].shift(1)
        triples = out[['range_lag2', 'range_lag1', 'normalized_range']].to_numpy()
        valid = np.isfinite(triples).all(axis=1)
        # Stable sorting resolves ties by chronological order.
        order = np.argsort(triples, axis=1, kind='stable')
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out['pattern'] = pd.Series(codes, index=data.index).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = out['pattern'].eq(code).astype(float).where(valid)
            probability = occurrence.rolling(
                params.entropy_window, min_periods=params.entropy_window).mean()
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out['entropy'] = (entropy / np.log(6.0)).clip(0.0, 1.0)

        body_low = data[['open', 'close']].min(axis=1, skipna=False)
        body_high = data[['open', 'close']].max(axis=1, skipna=False)
        out['lower_wick'] = (body_low - data['low']).clip(lower=0)
        out['upper_wick'] = (data['high'] - body_high).clip(lower=0)
        out['wick_pressure'] = ((out['lower_wick'] - out['upper_wick']) / width).clip(-1, 1)
        out['pressure'] = out['wick_pressure'].rolling(
            params.pressure_window, min_periods=params.pressure_window).mean()
        out['strength'] = (out['pressure'].abs() * (1 - out['entropy'])
                           * (1 - out['compression']).clip(0, 1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangePatternParams) -> SignalFrame:
        ready = indicators[['compression', 'entropy', 'pressure', 'strength']].notna().all(axis=1)
        gate = (ready & (indicators['compression'] <= params.compression_limit)
                & (indicators['entropy'] <= params.entropy_limit))
        candidate = np.select(
            [gate & (indicators['pressure'] >= params.pressure_threshold),
             gate & (indicators['pressure'] <= -params.pressure_threshold)],
            [1, -1], default=0).astype(int)
        conviction = (indicators['strength'].fillna(0) / params.conviction_scale).clip(0, 1)
        entry_size = (0.20 + 0.80 * conviction).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        position = 0
        held_size = 0.20
        for i, direction in enumerate(candidate):
            if position == 0 and direction != 0:
                position = int(direction)
                held_size = float(entry_size[i])
            elif position != 0 and direction == -position:
                # Opposite qualifying entry closes first; no same-bar reversal.
                position = 0
                held_size = 0.20
            signals[i] = position
            sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RangePatternParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    pressure = pd.Series([0.3, 0.0, 0.3, -0.3, -0.3, 0.0, 0.3, 0.3], index=index)
    indicators = pd.DataFrame({'compression': 0.7, 'entropy': 0.4,
                               'pressure': pressure, 'strength': 0.04}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert np.isclose(result['size'].iloc[1], 0.6)
    assert (result['size'] > 0).all()
    assert result.index.equals(index)


if __name__ == '__main__':
    _self_check()
