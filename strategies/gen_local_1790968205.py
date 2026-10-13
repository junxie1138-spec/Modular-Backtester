from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 5
    regime_window: int = 20
    pressure_window: int = 3
    atr_window: int = 14
    area_threshold: float = 0.12
    efficiency_threshold: float = 0.30
    pressure_threshold: float = 0.15
    atr_multiple: float = 1.2
    hold_bars: int = 2

    def __post_init__(self):
        for value in (self.area_window, self.regime_window,
                      self.pressure_window, self.atr_window, self.hold_bars):
            if type(value) is not int or value < 1:
                raise ValueError('Window lengths and holding period must be positive integers')
        if self.area_window < 2:
            raise ValueError('area_window must be at least two')
        for value in (self.area_threshold, self.efficiency_threshold,
                      self.pressure_threshold):
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError('Thresholds must be finite and strictly between zero and one')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0.0:
            raise ValueError('atr_multiple must be finite and positive')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790968205'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(params.area_window + 1, params.regime_window + 1,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data, params):
        close = data['close'].astype(float).where(data['close'] > 0)
        volume = data['volume'].astype(float).where(data['volume'] > 0)
        price_step = close.pct_change(fill_method=None)
        volume_step = np.log(volume).diff()
        out = pd.DataFrame(index=data.index)
        out['price_step'] = price_step
        out['volume_step'] = volume_step

        # Antisymmetric second-order signature of cumulative price/volume steps.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.area_window):
            cross = price_step.shift(lag) * volume_step - volume_step.shift(lag) * price_step
            area = area + cross.rolling(params.area_window - lag).sum()
        scale = (price_step.abs().rolling(params.area_window).sum()
                 * volume_step.abs().rolling(params.area_window).sum())
        out['area'] = (area / scale.where(scale > 0)).clip(-1.0, 1.0)
        path_length = price_step.abs().rolling(params.regime_window).sum()
        out['efficiency'] = (price_step.rolling(params.regime_window).sum().abs()
                             / path_length.where(path_length > 0))

        spread = data['high'] - data['low']
        location = ((2.0 * close - data['high'] - data['low'])
                    / spread.where(spread > 0)).clip(-1.0, 1.0)
        location = location.mask(spread.eq(0), 0.0)
        out['close_location'] = location
        out['pressure'] = ((location * volume).rolling(params.pressure_window).sum()
                           / volume.rolling(params.pressure_window).sum())
        previous_close = close.shift(1)
        true_range = pd.concat([spread, (data['high'] - previous_close).abs(),
                                (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['true_range'] = true_range
        out['atr'] = true_range.rolling(params.atr_window).mean()
        regime = pd.Series(np.where(out['efficiency'] >= params.efficiency_threshold,
                                    1.0, -1.0), index=data.index)
        out['regime'] = regime.where(out['efficiency'].notna())
        out['directional_area'] = out['area'] * out['regime']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        required = indicators[['directional_area', 'pressure', 'atr', 'efficiency']]
        valid = pd.Series(np.isfinite(required.to_numpy(dtype=float)).all(axis=1),
                          index=data.index)
        valid &= indicators['atr'].gt(0)
        valid &= data['close'].gt(0) & np.isfinite(data['close'])
        buy = (valid & indicators['directional_area'].gt(params.area_threshold)
               & indicators['pressure'].gt(params.pressure_threshold))
        sell = (valid & indicators['directional_area'].lt(-params.area_threshold)
                & indicators['pressure'].lt(-params.pressure_threshold))
        candidates = np.select([buy, sell], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        usable = valid.to_numpy(dtype=bool)
        position = 0
        held = 0
        high_water = low_water = np.nan
        floor = -np.inf
        ceiling = np.inf

        for i in range(len(data)):
            if position:
                # A decision on the preceding close entered at this session's open.
                held += 1
                if not usable[i]:
                    position = 0
                else:
                    if held == 1:
                        high_water = low_water = closes[i]
                    else:
                        high_water = max(high_water, closes[i])
                        low_water = min(low_water, closes[i])
                    floor = max(floor, high_water - params.atr_multiple * atrs[i])
                    # The required high-water exit applies to both directions.
                    # Shorts additionally have a downward-ratcheting adverse-price stop.
                    ceiling = min(ceiling, low_water + params.atr_multiple * atrs[i])
                    stopped = closes[i] < floor
                    if position == -1:
                        stopped = stopped or closes[i] > ceiling
                    if stopped or held >= params.hold_bars:
                        position = 0
                # An exit emits zero; it cannot reopen on the same decision bar.
            elif usable[i] and candidates[i] != 0:
                position = int(candidates[i])
                held = 0
                floor = -np.inf
                ceiling = np.inf
            decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': [100.0] * 6,
                         'high': [101.0] * 6,
                         'low': [97.0] * 6,
                         'close': [100.0, 100.0, 98.0, 98.0, 98.0, 98.0],
                         'volume': [1000.0] * 6}, index=index)
    supplied = pd.DataFrame({'directional_area': [0.5, 0, 0, 0, 0, 0],
                             'pressure': [0.5, 0, 0, 0, 0, 0],
                             'atr': [0.5] * 6,
                             'efficiency': [0.5] * 6}, index=index)
    # A three-session test isolates the trailing exit from the two-session default expiry.
    result = strategy.generate_signals(data, supplied, None, SignatureParams(hold_bars=3))
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    supplied['pressure'] = -0.5
    assert not strategy.generate_signals(data, supplied, None, SignatureParams()).data['signal'].any()
    supplied['directional_area'] = [-0.5, 0, 0, 0, 0, 0]
    data['close'] = 100.0
    result = strategy.generate_signals(data, supplied, None, SignatureParams())
    assert result.data['signal'].tolist() == [0, -1, -1, 0, 0, 0]
    assert result.data['size'].gt(0).all()
