from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 20
    pressure_window: int = 8
    max_damage_share: float = 0.35
    trailing_atr: float = 0.9
    hold_bars: int = 2

    def __post_init__(self):
        if self.peak_window < 2 or self.pressure_window < 3:
            raise ValueError('Lookbacks must accommodate the signal history.')
        if not 0.0 < self.max_damage_share < 1.0:
            raise ValueError('max_damage_share must be between zero and one.')
        if not np.isfinite(self.trailing_atr) or self.trailing_atr <= 0.0:
            raise ValueError('trailing_atr must be finite and positive.')
        if self.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791336125'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.peak_window, params.pressure_window + 1, 15))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['return'] = close.div(out['previous_close'].where(out['previous_close'] > 0)).sub(1)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - close.div(out['peak'].where(out['peak'] > 0))
        out['selling_budget'] = (-out['return'].clip(upper=0)).rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        losses = out['return'].lt(0).astype(float).where(out['return'].notna())
        out['losing_sessions'] = losses.rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['net_return'] = close.div(close.shift(params.pressure_window).where(close.shift(params.pressure_window) > 0)).sub(1)
        out['damage_share'] = out['drawdown'].div(out['selling_budget'].where(out['selling_budget'] > 0))
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            close.gt(out['ma200'])
            & out['return'].lt(0)
            & out['drawdown'].gt(0)
            & out['damage_share'].le(params.max_damage_share)
            & out['losing_sessions'].ge(3)
            & out['net_return'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        regimes = indicators['ma200'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr, regime = closes[i], atrs[i], regimes[i]
            valid = np.isfinite(close) and np.isfinite(atr) and np.isfinite(regime) and close > 0 and atr > 0
            if active:
                age += 1
                if valid:
                    # Entry fills on this bar; only its close and later closes enter the high-water mark.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                if not valid or close <= regime or close <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif valid and close > regime and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100., 105., 106., 103., 104.],
        'high': [101., 106., 107., 104., 105.],
        'low': [99., 104., 105., 102., 103.],
        'close': [100., 105., 106., 103., 104.],
        'volume': 1000.,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1., 1., 10., 10., 1.],
        'ma200': 90.,
        'entry': [True, False, False, False, False],
    }, index=index)
    expiry = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams())
    assert expiry.data['signal'].tolist() == [0, 1, 1, 0, 0]
    trailing = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams(hold_bars=10))
    assert trailing.data['signal'].tolist() == [0, 1, 1, 1, 0]
    assert trailing.data['size'].gt(0).all()
    indicators['ma200'] = 110.
    filtered = GeneratedStrategy.generate_signals(data, indicators, None, StrategyParams())
    assert filtered.data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
