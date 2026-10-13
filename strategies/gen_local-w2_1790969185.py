from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 12
    dominance: float = 0.60
    holding_bars: int = 18

    def __post_init__(self):
        if not isinstance(self.energy_window, int) or self.energy_window < 2:
            raise ValueError('energy_window must be an integer >= 2')
        if not 0.5 < self.dominance < 1.0:
            raise ValueError('dominance must be between 0.5 and 1')
        if not isinstance(self.holding_bars, int) or self.holding_bars < 1:
            raise ValueError('holding_bars must be a positive integer')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local-w2_1790969185'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params):
        return int(16 + params.energy_window)

    def indicators(self, data, params):
        f = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        f['log_close'] = np.log(close).replace([np.inf, -np.inf], np.nan)
        f['return'] = f['log_close'].diff()
        for half in (1, 2, 4, 8):
            width = 2 * half
            f[f'block_{half}'] = f['return'].rolling(half, min_periods=half).sum()
            f[f'haar_{width}'] = (
                f[f'block_{half}'] - f[f'block_{half}'].shift(half)
            ) / np.sqrt(width)
            f[f'energy_{width}'] = f[f'haar_{width}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        f['slow'] = f['return'].rolling(16, min_periods=16).sum() / 4.0
        f['slow_energy'] = f['slow'].pow(2).rolling(
            params.energy_window, min_periods=params.energy_window
        ).mean()
        f['fine_energy'] = f['energy_2'] + f['energy_4']
        f['coarse_energy'] = f['energy_8'] + f['energy_16'] + f['slow_energy']
        f['total_energy'] = f['fine_energy'] + f['coarse_energy']
        denominator = f['total_energy'].where(f['total_energy'] > 0)
        f['coarse_share'] = f['coarse_energy'] / denominator
        f['fine_share'] = f['fine_energy'] / denominator
        f['pullback'] = f['return'].rolling(4, min_periods=4).sum()
        f['continuation'] = (
            (f['coarse_share'] >= params.dominance)
            & (f['slow'] > 0)
            & (f['haar_16'] > 0)
        ).fillna(False)
        f['recovery'] = (
            (f['fine_share'] >= params.dominance)
            & (f['pullback'] < 0)
            & (f['return'] > 0)
            & (f['haar_2'] > 0)
        ).fillna(False)
        f['entry'] = f['continuation'] | f['recovery']
        f['strength'] = f['coarse_share'].where(
            f['continuation'], f['fine_share']
        ).fillna(0.0).clip(0.0, 1.0)
        f['entry_size'] = (
            0.25 + 0.70 * ((f['strength'] - params.dominance)
                           / (1.0 - params.dominance)).clip(0.0, 1.0)
        )
        return f

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.25, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        proposed_size = indicators['entry_size'].fillna(0.25).clip(0.25, 0.95).to_numpy()
        exit_at = -1
        held_size = 0.25
        for i in range(n):
            if i < exit_at:
                signal[i] = 1
                size[i] = held_size
            elif i == exit_at:
                # Emit a flat bar even if another entry qualifies today.
                size[i] = held_size
            elif entries[i]:
                exit_at = i + params.holding_bars
                held_size = float(proposed_size[i])
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = WaveletParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=80, freq='B')
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    synthetic = pd.DataFrame({'entry': True, 'entry_size': np.linspace(0.3, 0.9, len(index))}, index=index)
    out = strategy.generate_signals(data, synthetic, None, params).data
    assert out['signal'].iloc[0] == 0
    assert out['signal'].iloc[1:19].eq(1).all()
    assert out['signal'].iloc[19] == 0
    assert out['signal'].iloc[20] == 1
    assert out['size'].iloc[1:19].eq(0.3).all()
    assert out['size'].iloc[20] > out['size'].iloc[1]
    synthetic.loc[index[1]:, 'entry'] = False
    out = strategy.generate_signals(data, synthetic, None, params).data
    assert out['signal'].iloc[1:19].eq(1).all()
    assert out['signal'].iloc[19:].eq(0).all()
    flat = strategy.indicators(data, params)
    assert not flat['entry'].any()
    assert strategy.generate_signals(data, flat, None, params).data['signal'].eq(0).all()
    data['close'] = 100.0 * np.exp(np.cumsum(0.002 + 0.01 * np.sin(np.arange(80))))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:55], params)
    pd.testing.assert_frame_equal(full.iloc[:55], prefix)
    assert full['slow_energy'].first_valid_index() == index[strategy.warmup_bars(params) - 1]


if __name__ == '__main__':
    _self_check()
