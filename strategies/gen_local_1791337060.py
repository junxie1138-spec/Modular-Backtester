from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entry_z: float = 1.0
    trail_atr: float = 0.8

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791337060'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        # Ten closes, one difference, one lag, then twenty paired observations.
        return 31

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(10, min_periods=10).mean()
        out['price_std'] = close.rolling(10, min_periods=10).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].replace(0.0, np.nan)
        out['delta'] = out['z'].diff()
        out['lag_delta'] = out['delta'].shift(1)
        out['delta_mean'] = out['delta'].rolling(20, min_periods=20).mean()
        out['lag_mean'] = out['lag_delta'].rolling(20, min_periods=20).mean()
        out['delta_var'] = out['delta'].rolling(20, min_periods=20).var(ddof=0)
        out['lag_var'] = out['lag_delta'].rolling(20, min_periods=20).var(ddof=0)
        out['covariance'] = out['delta'].rolling(20, min_periods=20).cov(out['lag_delta'], ddof=0)
        out['beta'] = out['covariance'] / out['lag_var'].replace(0.0, np.nan)
        out['forecast'] = out['delta_mean'] + out['beta'] * (out['delta'] - out['lag_mean'])
        out['noise'] = np.sqrt((out['delta_var'] - out['beta'] * out['covariance']).clip(lower=0.0))
        out['previous_close'] = close.shift(1)
        out['range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['range', 'high_gap', 'low_gap']].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ['z', 'delta', 'beta', 'forecast', 'noise', 'atr']
        valid = pd.Series(np.isfinite(indicators[required]).all(axis=1), index=data.index)
        entry = (
            valid
            & (indicators['z'] <= -params.entry_z)
            & (indicators['delta'] < 0.0)
            & (indicators['beta'] < 0.0)
            & (indicators['noise'] > 0.0)
            & (indicators['forecast'] > 0.25 * indicators['noise'])
            & (indicators['atr'] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0.0:
                active = False
                age = 0
                high_water = np.nan
                stop = -np.inf
                continue
            if active:
                age += 1
                # The first in-trade close is on the bar after the entry decision.
                high_water = close if np.isnan(high_water) else max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= 2:
                    active = False
                    age = 0
                    high_water = np.nan
                    stop = -np.inf
                else:
                    decisions[i] = 1
                # An exit cannot reopen the position on the same decision bar.
                continue
            if entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = Params()
    index = pd.date_range('2020-01-01', periods=7)
    close = np.array([100.0, 100.0, 97.0, 100.0, 100.0, 103.0, 104.0])
    data = pd.DataFrame({'open': close, 'high': close + 1.0,
                         'low': close - 1.0, 'close': close,
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'z': [-2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
                               'delta': -1.0, 'beta': -0.5,
                               'forecast': 1.0, 'noise': 1.0,
                               'atr': 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    flat = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0,
                         'close': 100.0, 'volume': 1000.0},
                        index=pd.date_range('2020-01-01', periods=50))
    derived = GeneratedStrategy.indicators(None, flat, params)
    result = GeneratedStrategy.generate_signals(None, flat, derived, None, params).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
