from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    coarse_share: float = 0.65
    efficiency_window: int = 6
    efficiency_threshold: float = 0.35
    atr_window: int = 14
    atr_multiple: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if self.energy_window < 2 or self.efficiency_window < 2 or self.atr_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.coarse_share < 1.0:
            raise ValueError('coarse_share must be between zero and one.')
        if not 0.0 < self.efficiency_threshold <= 1.0:
            raise ValueError('efficiency_threshold must be positive and at most one.')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0.0:
            raise ValueError('atr_multiple must be finite and positive.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791210270'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params):
        return int(max(params.energy_window + 7, params.efficiency_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0.0))
        for scale in (1, 2, 4):
            out[f'mean_{scale}'] = out['log_close'].rolling(scale, min_periods=scale).mean()
            out[f'haar_{scale}'] = np.sqrt(scale / 2.0) * (
                out[f'mean_{scale}'] - out[f'mean_{scale}'].shift(scale)
            )
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out['total_energy'] = out[['energy_1', 'energy_2', 'energy_4']].sum(axis=1, min_count=3)
        out['coarse_fraction'] = out['energy_4'] / out['total_energy'].where(out['total_energy'] > 0.0)
        out['log_return'] = out['log_close'].diff()
        out['net_move'] = out['log_return'].rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out['path_length'] = out['log_return'].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out['efficiency'] = out['net_move'] / out['path_length'].where(out['path_length'] > 0.0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['wavelet_direction'] = np.where(
            (out['coarse_fraction'] >= params.coarse_share)
            & (out['haar_4'] * out['haar_2'] > 0.0),
            np.sign(out['haar_4']), 0
        ).astype(int)
        out['efficiency_direction'] = np.where(
            out['efficiency'] >= params.efficiency_threshold, 1,
            np.where(out['efficiency'] <= -params.efficiency_threshold, -1, 0)
        ).astype(int)
        out['entry_direction'] = np.where(
            out['wavelet_direction'] == out['efficiency_direction'],
            out['wavelet_direction'], 0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or close <= 0.0 or not np.isfinite(atr) or atr <= 0.0:
                position = 0
                continue
            if position:
                age += 1
                # Initialize extrema on the first close after the entry decision.
                high_water = close if np.isnan(high_water) else max(high_water, close)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                high_stop = max(high_stop, high_water - params.atr_multiple * atr)
                if position == -1:
                    short_stop = min(short_stop, low_water + params.atr_multiple * atr)
                # The required close-high floor applies to both directions.
                # Shorts additionally have a mirrored low-water rebound stop.
                if close < high_stop or (position == -1 and close > short_stop) or age >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                low_water = np.nan
                high_stop = -np.inf
                short_stop = np.inf
            targets[i] = position
        df = pd.DataFrame({'signal': targets, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = WaveletParams(atr_multiple=1.0, hold_bars=5)
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 100.0, 103.0, 102.5, 101.5, 101.0]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 1.0, 5.0, 5.0, 1.0],
        'entry_direction': [1, 0, 0, 0, 0, 0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    # ATR expansion cannot lower the previously established stop of 102.
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
