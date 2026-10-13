from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 8
    oversold_z: float = 1.25
    memory_bars: int = 18

    def __post_init__(self):
        if not 2 <= self.lookback <= 9:
            raise ValueError('lookback must be between 2 and 9')
        if not np.isfinite(self.oversold_z) or self.oversold_z <= 0:
            raise ValueError('oversold_z must be finite and positive')
        if self.memory_bars < 2:
            raise ValueError('memory_bars must be at least 2')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791204800'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        mean = close.rolling(params.lookback, min_periods=params.lookback).mean()
        dispersion = close.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        scale = dispersion.replace(0.0, np.nan)
        price_change = close.diff()
        ma_change = mean.diff()
        distance_z = (close - mean) / scale
        recovery_impulse = (price_change - ma_change.clip(lower=0.0)) / scale.shift(1)
        valid = np.isfinite(distance_z) & np.isfinite(recovery_impulse)
        result = pd.DataFrame({
            'mean': mean,
            'dispersion': dispersion,
            'price_change': price_change,
            'ma_change': ma_change,
            'distance_z': distance_z,
            'recovery_impulse': recovery_impulse,
        }, index=data.index)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result['valid'] = valid.fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        retention = 1.0 - 1.0 / params.memory_bars
        armed = False
        holding = False
        balance = 0.0
        decisions = np.zeros(len(data), dtype=int)
        balances = np.zeros(len(data), dtype=float)
        rows = zip(indicators['distance_z'], indicators['recovery_impulse'],
                   indicators['valid'])
        for i, (distance_z, impulse, valid) in enumerate(rows):
            if valid:
                if not armed and distance_z <= -params.oversold_z:
                    armed = True
                    balance = 0.0
                if armed:
                    balance = retention * balance + impulse
                    # The bullish condition is a strictly positive recovery balance.
                    # Its reversal is the only exit; there is no timed liquidation.
                    if holding and balance <= 0.0:
                        armed = False
                        balance = 0.0
                    elif not holding:
                        balance = max(0.0, balance)
                holding = armed and balance > 0.0
            decisions[i] = int(holding)
            balances[i] = balance
        df = pd.DataFrame({'signal': decisions, 'size': 1.0,
                           'recovery_balance': balances}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    close = np.array([100.0] * 10 + [99, 98, 97, 96, 95, 96, 97, 98, 99,
                                     100, 101, 102, 101, 99, 97, 95, 93, 91, 89, 87])
    data = pd.DataFrame({'open': close, 'high': close + 1.0,
                         'low': close - 1.0, 'close': close, 'volume': 1000.0},
                        index=pd.date_range('2020-01-01', periods=len(close)))
    strategy = object.__new__(GeneratedStrategy)
    params = RecoveryParams()
    features = strategy.indicators(data, params)
    result = strategy.generate_signals(data, features, None, params).data
    expected = (result['recovery_balance'] > 0.0).astype(int)
    expected = expected.shift(1).fillna(0).astype(int)
    assert np.array_equal(result['signal'].to_numpy(), expected.to_numpy())
    assert set(result['signal'].unique()) == {0, 1}
    assert (result['signal'].diff() == -1).any()
    assert (result['size'] > 0.0).all()
    assert strategy.warmup_bars(params) <= 10
    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc('close')] *= 2.0
    altered = strategy.generate_signals(changed, strategy.indicators(changed, params),
                                        None, params).data
    assert np.array_equal(result['signal'].to_numpy(), altered['signal'].to_numpy())
    flat = data.copy()
    flat['close'] = 100.0
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params),
                                            None, params).data
    assert flat_result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
