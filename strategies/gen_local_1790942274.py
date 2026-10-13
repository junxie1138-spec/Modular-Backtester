from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPressureParams:
    energy_window: int = 3
    energy_ratio: float = 1.5
    pressure_window: int = 4
    pressure_threshold: float = 0.15
    atr_window: int = 14
    stop_atr: float = 1.8
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.energy_window, self.pressure_window, self.atr_window) < 1:
            raise ValueError('Lookbacks must be positive')
        if not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be between 3 and 5')
        if not np.isfinite(self.energy_ratio) or self.energy_ratio <= 0:
            raise ValueError('energy_ratio must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not np.isfinite(self.pressure_threshold) or not 0 < self.pressure_threshold < 1:
            raise ValueError('pressure_threshold must be between 0 and 1')


class GeneratedStrategy(BaseStrategy[WaveletPressureParams]):
    strategy_id = 'gen_local_1790942274'

    @classmethod
    def params_type(cls):
        return WaveletPressureParams

    def warmup_bars(self, params):
        return int(max(8 + params.energy_window, params.atr_window + 1, params.pressure_window))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['log_return'] = np.log(close).diff()
        r = out['log_return']
        out['haar_detail_2'] = (r - r.shift(1)) / np.sqrt(2.0)
        pair = r.rolling(2, min_periods=2).sum()
        out['haar_detail_4'] = (pair - pair.shift(2)) / 2.0
        out['haar_approx_8'] = r.rolling(8, min_periods=8).sum() / np.sqrt(8.0)
        w = params.energy_window
        out['coarse_energy'] = out['haar_approx_8'].pow(2).rolling(w, min_periods=w).mean()
        out['detail_energy'] = (out['haar_detail_2'].pow(2) + out['haar_detail_4'].pow(2)).rolling(w, min_periods=w).mean()
        out['energy_ratio'] = out['coarse_energy'] / out['detail_energy'].clip(lower=1e-12)
        spread = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / spread).clip(-1.0, 1.0)
        out.loc[data['high'].eq(data['low']), 'close_location'] = 0.0
        volume = data['volume'].where(data['volume'] >= 0)
        out['signed_volume'] = out['close_location'] * volume
        p = params.pressure_window
        out['pressure_volume'] = volume.rolling(p, min_periods=p).sum()
        out['pressure'] = out['signed_volume'].rolling(p, min_periods=p).sum() / out['pressure_volume'].where(out['pressure_volume'] > 0)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        ready = indicators[['energy_ratio', 'haar_approx_8', 'pressure', 'atr']].notna().all(axis=1)
        ready &= np.isfinite(data['close']) & data['close'].gt(0) & indicators['atr'].gt(0)
        energetic = indicators['energy_ratio'].ge(params.energy_ratio)
        long_entry = ready & energetic & indicators['haar_approx_8'].gt(0) & indicators['pressure'].ge(params.pressure_threshold)
        short_entry = ready & energetic & indicators['haar_approx_8'].lt(0) & indicators['pressure'].le(-params.pressure_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        decision_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if direction:
                if i == decision_bar + 1:
                    # Entry becomes observable on the next bar; ATR stays frozen at the decision bar.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        direction = 0
                        continue
                    stop = entry_price - direction * params.stop_atr * entry_atr
                stopped = np.isfinite(closes[i]) and direction * (closes[i] - stop) <= 0
                expired = i - decision_bar >= params.holding_bars
                if stopped or expired or not np.isfinite(closes[i]):
                    direction = 0
                    continue
                decisions[i] = direction
            elif candidates[i]:
                direction = int(candidates[i])
                decision_bar = i
                entry_atr = atr[i]
                entry_price = np.nan
                stop = np.nan
                decisions[i] = direction
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        close = np.full(8, 100.0)
        close[2] = 100.0 - direction * 3.0
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'energy_ratio': 3.0, 'haar_approx_8': direction * 0.01, 'pressure': direction * 0.5, 'atr': 1.0}, index=index)
        stopped = strategy.generate_signals(data, derived, None, WaveletPressureParams(stop_atr=1.0)).data
        assert stopped['signal'].iloc[0] == 0
        assert stopped['signal'].iloc[1] == direction
        assert stopped['signal'].iloc[2] == direction
        assert stopped['signal'].iloc[3] == 0
        timed = strategy.generate_signals(data, derived, None, WaveletPressureParams(stop_atr=100.0)).data
        assert timed['signal'].iloc[1:5].eq(direction).all()
        assert timed['signal'].iloc[5] == 0
        assert stopped['size'].gt(0).all()
    params = WaveletPressureParams()
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    flat['volume'] = 0.0
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
