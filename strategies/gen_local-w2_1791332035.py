from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    entry_z: float = 1.5
    trail_atr: float = 2.0

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791332035'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return 96

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0))
        weights = np.ones(32)
        for lag in range(1, 32):
            weights[lag] = -weights[lag - 1] * (0.45 - lag + 1) / lag
        terms = pd.concat(
            [out['log_close'].shift(lag) * weights[lag] for lag in range(32)],
            axis=1,
        )
        out['fd'] = terms.sum(axis=1, min_count=32)
        out['fd_mean'] = out['fd'].rolling(60, min_periods=60).mean()
        out['fd_std'] = out['fd'].rolling(60, min_periods=60).std(ddof=0)
        out['fd_z'] = (out['fd'] - out['fd_mean']) / out['fd_std'].where(out['fd_std'] > 0)
        out['fd_change'] = out['fd'].diff()
        previous_close = close.shift(1)
        out['tr'] = pd.concat(
            [data['high'] - data['low'],
             (data['high'] - previous_close).abs(),
             (data['low'] - previous_close).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out['atr'] = out['tr'].rolling(14, min_periods=14).mean()
        out['setup'] = (
            (out['fd_z'].shift(1) < -params.entry_z)
            & (out['fd_z'] < -0.5)
            & (out['fd_change'] > 0)
            & (out['fd_change'].shift(1) <= 0)
            & (out['atr'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        armed = False
        age = 0
        setup = indicators['setup'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)

        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite([opening, high, low, close, atr[i]]).all()
            valid = valid and min(opening, high, low, close, atr[i]) > 0
            if not valid:
                active = False
                entry = np.nan
                continue

            if active:
                if np.isnan(entry):
                    # The previous close's entry decision fills at this open.
                    entry = opening
                    peak = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                # Test the existing stop before raising it with today's high.
                exited = low <= stop
                if not exited:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * 1.015
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    exited = close <= stop or age >= 10
                if exited:
                    active = False
                    entry = np.nan
                    # Exit decisions execute at the following bar's open.
                    continue
            elif setup[i]:
                active = True
                entry = np.nan

            signals[i] = int(active)

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100, 100, 100.5, 102.8, 101, 101],
        'high': [100.5, 101, 103, 103, 102, 102],
        'low': [99.5, 99.5, 100, 100.8, 100, 100],
        'close': [100, 100.5, 102.8, 101, 101, 101],
        'volume': [1000] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 6,
        'setup': [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, StrategyParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result.data['size'] > 0).all()
