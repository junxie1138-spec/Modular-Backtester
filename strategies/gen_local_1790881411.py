from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryReleaseParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    atr_window: int = 14
    impulse_atr: float = 0.08
    stop_atr: float = 1.2

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if self.fractional_window < 2 or self.atr_window < 2:
            raise ValueError('indicator windows must be at least two')
        if not np.isfinite(self.impulse_atr) or self.impulse_atr < 0:
            raise ValueError('impulse_atr must be finite and nonnegative')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[MemoryReleaseParams]):
    strategy_id = 'gen_local_1790881411'

    @classmethod
    def params_type(cls):
        return MemoryReleaseParams

    def warmup_bars(self, params):
        return int(max(200, params.fractional_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        close = data['close'].astype(float)
        fractional = close.copy()
        weight = 1.0
        # Finite fractional filter: lag zero has weight one.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        change = close.diff()
        impulse = fractional.diff() / atr.where(atr > 0)
        out = pd.DataFrame({
            'fractional_close': fractional,
            'close_change': change,
            'atr': atr,
            'impulse': impulse,
            'previous_impulse': impulse.shift(1),
            'ma200': close.rolling(200, min_periods=200).mean(),
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = (data['close'] > indicators['ma200']).fillna(False).to_numpy()
        ready = indicators.notna().all(axis=1).to_numpy()
        enter = (
            (indicators['impulse'] > params.impulse_atr)
            & (indicators['previous_impulse'] > 0)
            & (indicators['close_change'] < 0)
        ).fillna(False).to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        frozen_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    fixed_stop = opening[i] - params.stop_atr * frozen_atr
                exit_now = (
                    not np.isfinite(close[i])
                    or not np.isfinite(fixed_stop)
                    or close[i] < fixed_stop
                    or not regime[i]
                    or i - entry_bar + 1 >= 2
                )
                if exit_now:
                    active = False
                else:
                    decisions[i] = 1
                continue
            if ready[i] and regime[i] and atr[i] > 0 and enter[i]:
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                fixed_stop = np.nan
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = MemoryReleaseParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 96.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'fractional_close': 10.0, 'close_change': -0.1,
        'atr': 2.0, 'impulse': 1.0, 'previous_impulse': 1.0,
        'ma200': 90.0,
    }, index=index)
    # These methods have no dependency on BaseStrategy initialization.
    signals = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert signals['signal'].tolist() == [0, 1, 1, 0, 1]
    data.loc[index[0], 'close'] = 110.0
    data.loc[index[1], 'close'] = 98.0
    signals = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert signals['signal'].iloc[2] == 1  # Stop anchors to entry open, not decision close.
    data.loc[index[1], 'close'] = 97.0
    signals = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert signals['signal'].iloc[2] == 0
    assert signals['signal'].isin([0, 1]).all()
    assert (signals['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
