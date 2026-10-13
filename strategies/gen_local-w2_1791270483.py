from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveEnergyParams:
    energy_window: int = 6
    baseline_window: int = 10
    fast_decay: float = 0.65
    slow_share: float = 0.60
    pressure_window: int = 3
    pressure_threshold: float = 0.15
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    atr_multiple: float = 2.0
    breakeven_gain: float = 0.008
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[WaveEnergyParams]):
    strategy_id = "gen_local-w2_1791270483"

    @classmethod
    def params_type(cls) -> type[WaveEnergyParams]:
        return WaveEnergyParams

    def warmup_bars(self, params: WaveEnergyParams) -> int:
        return int(max(8 + params.energy_window + params.baseline_window + 1,
                       params.volatility_window + 1, params.atr_window + 1,
                       params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: WaveEnergyParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['returns'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        for half in (1, 2, 4):
            recent = out['returns'].rolling(half, min_periods=half).sum()
            out[f'haar_{2 * half}'] = (recent - recent.shift(half)) / np.sqrt(2.0 * half)
            out[f'energy_{2 * half}'] = out[f'haar_{2 * half}'].pow(2).rolling(
                p.energy_window, min_periods=p.energy_window).mean()
        out['fast_reference'] = out['energy_2'].shift(1).rolling(
            p.baseline_window, min_periods=p.baseline_window).mean()
        out['total_energy'] = out[['energy_2', 'energy_4', 'energy_8']].sum(axis=1, min_count=3)
        out['slow_fraction'] = (out['energy_4'] + out['energy_8']) / out['total_energy'].where(
            out['total_energy'] > 0)
        span = data['high'] - data['low']
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) /
                                 span.where(span > 0)).clip(-1.0, 1.0).fillna(0.0)
        volume = data['volume'].clip(lower=0).fillna(0.0)
        out['signed_volume'] = out['close_location'] * volume
        out['pressure_volume'] = volume.rolling(p.pressure_window, min_periods=p.pressure_window).sum()
        out['pressure'] = out['signed_volume'].rolling(
            p.pressure_window, min_periods=p.pressure_window).sum() / out['pressure_volume'].where(
                out['pressure_volume'] > 0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([span, (data['high'] - previous_close).abs(),
                                      (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out['volatility'] = out['returns'].rolling(
            p.volatility_window, min_periods=p.volatility_window).std(ddof=0) * np.sqrt(252.0)
        out['target_size'] = (p.target_volatility / out['volatility'].where(
            out['volatility'] > 0)).clip(0.10, 0.95).fillna(0.10)
        out['wave_agreement'] = ((out['fast_reference'] > 0) &
                                 (out['energy_2'] < p.fast_decay * out['fast_reference']) &
                                 (out['slow_fraction'] >= p.slow_share))
        out['pressure_agreement'] = out['pressure'] > p.pressure_threshold
        out['entry_setup'] = (out['wave_agreement'] & out['pressure_agreement'] &
                              (out['atr'] > 0) & (out['volatility'] > 0)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveEnergyParams) -> SignalFrame:
        p = params
        raw = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.10, dtype=float)
        setup = indicators['entry_setup'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        target_size = indicators['target_size'].fillna(0.10).clip(0.10, 0.95).to_numpy(dtype=float)
        active = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        entry_atr = np.nan
        armed = False
        age = 0
        held_size = 0.10
        for i, (op, hi, lo, cl) in enumerate(data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)):
            valid_bar = np.isfinite([op, hi, lo, cl]).all() and min(op, hi, lo, cl) > 0
            if active:
                if not valid_bar:
                    active = False
                else:
                    # The previous close's entry instruction fills at this open.
                    if not np.isfinite(entry):
                        entry = op
                        peak = entry
                        stop = entry - p.atr_multiple * entry_atr
                    age += 1
                    # Test the previously established stop before updating it.
                    if lo <= stop:
                        active = False
                    else:
                        peak = max(peak, hi)
                        armed = armed or peak >= entry * (1.0 + p.breakeven_gain)
                        if armed:
                            stop = max(stop, entry)
                            if np.isfinite(atr[i]) and atr[i] > 0:
                                stop = max(stop, peak - p.atr_multiple * atr[i])
                        # Newly raised stops are checked at the close; exits fill next bar.
                        if cl <= stop or age >= p.holding_bars:
                            active = False
                raw[i] = int(active)
                sizes[i] = held_size
                continue
            if valid_bar and setup[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry = np.nan
                peak = np.nan
                armed = False
                age = 0
                entry_atr = atr[i]
                held_size = target_size[i]
                raw[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({'signal': raw, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    p = WaveEnergyParams()
    bars = pd.DataFrame({'open': [100.0] * 6, 'high': [101.0] * 6,
                         'low': [99.5, 100.0, 99.9, 100.0, 100.0, 100.0],
                         'close': [100.5] * 6}, index=pd.date_range('2020-01-01', periods=6))
    features = pd.DataFrame({'entry_setup': [True, False, False, False, False, False],
                             'atr': [1.0] * 6, 'target_size': [0.5] * 6}, index=bars.index)
    result = GeneratedStrategy.generate_signals(None, bars, features, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert result['size'].gt(0).all()
    bars['high'] = 100.6
    bars['low'] = 99.9
    result = GeneratedStrategy.generate_signals(None, bars, features, None, p).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    for n in range(1, len(bars) + 1):
        prefix = GeneratedStrategy.generate_signals(None, bars.iloc[:n], features.iloc[:n], None, p).data
        pd.testing.assert_frame_equal(prefix, result.iloc[:n])
