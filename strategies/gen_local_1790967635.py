from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fast_window: int = 5
    slow_window: int = 40
    drift_window: int = 5
    atr_window: int = 14
    quiet_ratio: float = 0.85
    stress_ratio: float = 1.25
    entry_strength: float = 0.25
    stop_atr: float = 2.5
    max_hold: int = 8

    def __post_init__(self):
        if not (2 <= self.fast_window <= self.slow_window):
            raise ValueError('Require 2 <= fast_window <= slow_window')
        if min(self.drift_window, self.atr_window, self.max_hold) < 1:
            raise ValueError('Lookbacks and max_hold must be positive')
        if not (0 < self.quiet_ratio < 1 < self.stress_ratio):
            raise ValueError('Require quiet_ratio < 1 < stress_ratio')
        if not (0 < self.entry_strength <= 1 and 0 < self.stop_atr < np.inf):
            raise ValueError('Invalid entry strength or stop distance')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790967635'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(params.slow_window + 2, params.fast_window + 2,
                       params.drift_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['return'] = close.div(out['previous_close'].where(out['previous_close'] > 0)).sub(1)
        out['return'] = out['return'].replace([np.inf, -np.inf], np.nan)
        out['fast_vol'] = out['return'].rolling(params.fast_window).std(ddof=0)
        out['slow_vol'] = out['return'].rolling(params.slow_window).std(ddof=0)
        out['vol_ratio'] = out['fast_vol'].div(out['slow_vol'].where(out['slow_vol'] > 0))
        out['previous_ratio'] = out['vol_ratio'].shift(1)
        out['cooling'] = (1 - out['vol_ratio'].div(out['previous_ratio'].where(out['previous_ratio'] > 0))).clip(0, 1)
        out['drift'] = out['return'].rolling(params.drift_window).sum()
        out['directional_score'] = out['drift'].div(out['slow_vol'].where(out['slow_vol'] > 0) * np.sqrt(params.drift_window))
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['quiet_strength'] = out['directional_score'].clip(0, 1)
        out['stress_strength'] = (-out['directional_score']).clip(0, 1) * (10 * out['cooling']).clip(0, 1)
        out.loc[out['return'] <= 0, 'stress_strength'] = 0.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out[['quiet_strength', 'stress_strength']] = out[['quiet_strength', 'stress_strength']].fillna(0.0)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.25, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ratio = indicators['vol_ratio'].to_numpy(dtype=float)
        quiet = indicators['quiet_strength'].to_numpy(dtype=float)
        stressed = indicators['stress_strength'].to_numpy(dtype=float)
        regime = 0
        active = False
        entry_bar = -1
        highest_close = -np.inf
        stop = -np.inf
        entry_size = 0.25

        for i in range(count):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(ratio[i])
            if not valid:
                active = False
                regime = 0
                continue
            if ratio[i] <= params.quiet_ratio:
                regime = 1
            elif ratio[i] >= params.stress_ratio:
                regime = 2

            if active:
                # Entry fills on this or an earlier bar; exclude the decision-bar close.
                highest_close = max(highest_close, close[i])
                stop = max(stop, highest_close - params.stop_atr * atr[i])
                if close[i] <= stop or i - entry_bar >= params.max_hold:
                    active = False
                    continue
            else:
                strength = quiet[i] if regime == 1 else stressed[i] if regime == 2 else 0.0
                if not np.isfinite(strength) or strength < params.entry_strength:
                    continue
                active = True
                entry_bar = i
                highest_close = -np.inf
                stop = -np.inf
                entry_size = float(0.25 + 0.70 * np.clip(strength, 0.0, 1.0))

            signals[i] = 1
            sizes[i] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100., 103., 102., 101., 100., 100.]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1., 1., 5., 5., 5., 5.],
        'vol_ratio': [0.7] * 6,
        'quiet_strength': [1., 0., 0., 0., 0., 0.],
        'stress_strength': [0.] * 6,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    # Widening ATR must not lower the stop; exit and entry both shift one bar.
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert np.allclose(result.loc[result['signal'] == 1, 'size'], 0.95)
    indicators.loc[index[0], 'quiet_strength'] = 0.5
    smaller = strategy.generate_signals(data, indicators, None, params).data
    assert np.isclose(smaller['size'].iloc[1], 0.60)
