from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    efficiency_window: int = 6
    atr_window: int = 14
    entry_z: float = 1.3
    min_efficiency: float = 0.55
    stop_atr: float = 1.8
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.ma_window, self.efficiency_window, self.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive.')
        if not 0 < self.min_efficiency <= 1:
            raise ValueError('min_efficiency must be in (0, 1].')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791334416'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.ma_window, params.efficiency_window + 1,
                       params.atr_window + 1) + 1)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['std'].replace(0, np.nan)
        out['change'] = close.diff()
        out['path_length'] = out['change'].abs().rolling(params.efficiency_window).sum()
        out['net_change'] = close.diff(params.efficiency_window)
        out['efficiency'] = out['net_change'] / out['path_length'].replace(0, np.nan)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['candidate'] = (
            out['z'].ge(params.entry_z)
            & out['efficiency'].ge(params.min_efficiency)
            & out['change'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        out['confirmed'] = out['candidate'] & out['candidate'].shift(1, fill_value=False)
        out['entry'] = out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        peak = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    peak = max(peak, close)
                    stop = max(stop, peak - params.stop_atr * atr)
                if not valid or close <= stop or age >= params.hold_bars:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                # The first in-trade close arrives on the next bar after entry fills.
                peak = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [99., 100., 104., 103., 101., 100., 100.]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1., 1., 1., 10., 10., 10., 10.],
        'entry': [True, False, False, False, False, False, False]
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        data, indicators, None, StrategyParams(stop_atr=2., hold_bars=5)
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    flat = pd.DataFrame({'close': np.full(7, 100.)}, index=index)
    result = GeneratedStrategy.generate_signals(
        flat, indicators, None, StrategyParams(hold_bars=4)
    ).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    prices = pd.Series(np.linspace(100., 140., 60))
    bars = pd.DataFrame({'open': prices, 'high': prices + 1.,
                         'low': prices - 1., 'close': prices, 'volume': 1000.})
    derived = GeneratedStrategy.indicators(bars, StrategyParams())
    expected = derived['candidate'] & derived['candidate'].shift(1, fill_value=False)
    assert derived['confirmed'].equals(expected)
    assert derived['entry'].any()


if __name__ == '__main__':
    _self_check()
