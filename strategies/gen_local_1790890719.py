from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    gap_lookback: int = 40
    pressure_lookback: int = 5
    gap_threshold: float = 1.5
    pressure_threshold: float = 0.25
    holding_bars: int = 8

    def __post_init__(self):
        if self.gap_lookback < 2 or self.pressure_lookback < 1 or self.holding_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; gap lookback must exceed one.')
        if not np.isfinite(self.gap_threshold) or self.gap_threshold <= 0:
            raise ValueError('gap_threshold must be finite and positive.')
        if not np.isfinite(self.pressure_threshold) or not 0 < self.pressure_threshold <= 1:
            raise ValueError('pressure_threshold must be in (0, 1].')


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = 'gen_local_1790890719'

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return int(max(params.gap_lookback + 2, params.pressure_lookback))

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = data['open'] / out['prior_close'] - 1.0
        out['gap_scale'] = out['gap'].shift(1).rolling(params.gap_lookback).std(ddof=0)
        out['gap_z'] = out['gap'] / out['gap_scale'].where(out['gap_scale'] > 0)
        out['range'] = data['high'] - data['low']
        out['closing_pressure'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['range'].where(out['range'] > 0)).clip(-1.0, 1.0)
        out.loc[out['range'].eq(0) & data['close'].notna(), 'closing_pressure'] = 0.0
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0)
        out['weighted_pressure'] = out['closing_pressure'] * out['valid_volume']
        out['volume_sum'] = out['valid_volume'].rolling(params.pressure_lookback).sum()
        out['pressure_sum'] = out['weighted_pressure'].rolling(params.pressure_lookback).sum()
        out['pressure'] = out['pressure_sum'] / out['volume_sum'].where(out['volume_sum'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        long_entry = (indicators['gap_z'] <= -params.gap_threshold) & (indicators['pressure'] <= -params.pressure_threshold)
        short_entry = (indicators['gap_z'] >= params.gap_threshold) & (indicators['pressure'] >= params.pressure_threshold)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GapPressureParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=25, freq='B')
    data = pd.DataFrame(index=index)
    for direction in (1, -1):
        indicators = pd.DataFrame({'gap_z': -2.0 * direction, 'pressure': -0.5 * direction}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].iloc[0] == 0
        assert result['signal'].iloc[1:9].eq(direction).all()
        assert result['signal'].iloc[9] == 0
        assert result['signal'].iloc[10:18].eq(direction).all()
        assert result['signal'].iloc[18] == 0
        assert result['size'].gt(0).all()
    disagreement = pd.DataFrame({'gap_z': -2.0, 'pressure': 0.5}, index=index)
    assert strategy.generate_signals(data, disagreement, None, params).data['signal'].eq(0).all()
    missing = pd.DataFrame({'gap_z': np.nan, 'pressure': np.nan}, index=index)
    assert strategy.generate_signals(data, missing, None, params).data['signal'].eq(0).all()
