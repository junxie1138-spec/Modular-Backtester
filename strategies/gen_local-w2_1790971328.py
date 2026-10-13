from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolumeParams:
    seasonal_window: int = 60
    pressure_window: int = 7
    threshold: float = 0.30

    def __post_init__(self):
        if self.seasonal_window < 10 or self.pressure_window < 1:
            raise ValueError('Invalid lookback windows')
        if not 0.0 < self.threshold < 1.0:
            raise ValueError('threshold must be between zero and one')


class GeneratedStrategy(BaseStrategy[SeasonalVolumeParams]):
    strategy_id = 'gen_local-w2_1790971328'

    @classmethod
    def params_type(cls):
        return SeasonalVolumeParams

    def warmup_bars(self, params: SeasonalVolumeParams) -> int:
        return int(params.seasonal_window + params.pressure_window)

    def indicators(self, data: pd.DataFrame, params: SeasonalVolumeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['weekday'] = data.index.dayofweek
        out['volume'] = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['volume'] = out['volume'].where(out['volume'] >= 0.0)
        out['intraday_return'] = (data['close'] / data['open'].where(data['open'] > 0.0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['seasonal_volume'] = np.nan
        window = params.seasonal_window
        for weekday in range(7):
            same_day = out['weekday'].eq(weekday)
            valid = same_day & out['volume'].notna()
            total = out['volume'].where(valid, 0.0).rolling(window, min_periods=window).sum().shift(1)
            count = valid.astype(float).rolling(window, min_periods=window).sum().shift(1)
            name = 'weekday_baseline_' + str(weekday)
            out[name] = total / count.where(count > 0.0)
            out.loc[same_day, 'seasonal_volume'] = out.loc[same_day, name]
        out['relative_volume'] = out['volume'] / out['seasonal_volume'].where(out['seasonal_volume'] > 0.0)
        out['excess_volume'] = (out['relative_volume'] - 1.0).clip(lower=0.0)
        out['impulse'] = out['intraday_return'] * out['excess_volume']
        out['absolute_impulse'] = out['impulse'].abs()
        out['net_pressure'] = out['impulse'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['gross_pressure'] = out['absolute_impulse'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['pressure'] = (out['net_pressure'] / out['gross_pressure'].where(out['gross_pressure'] > 0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalVolumeParams) -> SignalFrame:
        pressure = indicators['pressure'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        entries = np.where(pressure >= params.threshold, 1, np.where(pressure <= -params.threshold, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # The mirrored entry condition closes the position for one bar.
                position = 0
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = SeasonalVolumeParams()
    index = pd.date_range('2020-01-01', periods=9, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'pressure': [np.nan, 0.8, 0.0, -0.8, -0.8, 0.0, 0.8, 0.8, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1, -1, 0, 1]
    mirrored = indicators.copy()
    mirrored['pressure'] = -mirrored['pressure']
    reverse = GeneratedStrategy.generate_signals(None, data, mirrored, None, params).data
    assert reverse['signal'].equals(-result['signal'])
    assert result['size'].gt(0.0).all()
