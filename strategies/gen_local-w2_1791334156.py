from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    ma_window: int = 20
    correlation_window: int = 63
    half_cycle: int = 10
    correlation_floor: float = 0.15
    entry_z: float = 0.9
    atr_window: int = 14
    atr_multiple: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = 'gen_local-w2_1791334156'

    @classmethod
    def params_type(cls) -> type[CycleParams]:
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(200, params.ma_window + 2 * params.half_cycle + params.correlation_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        if min(params.ma_window, params.correlation_window) < 2 or min(params.half_cycle, params.atr_window, params.holding_bars) < 1:
            raise ValueError('Invalid lookback or holding period')
        if not (0 < params.correlation_floor < 1 and params.entry_z > 0 and params.atr_multiple > 0):
            raise ValueError('Invalid entry or stop threshold')
        close = data['close']
        result = pd.DataFrame(index=data.index)
        result['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        result['std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        result['z'] = (close - result['ma']) / result['std'].replace(0.0, np.nan)
        result['regime_ma'] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        result['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        result['atr'] = result['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        z = result['z']
        result['half_lag_z'] = z.shift(params.half_cycle)
        result['full_lag_z'] = z.shift(2 * params.half_cycle)
        result['half_corr'] = z.rolling(params.correlation_window, min_periods=params.correlation_window).corr(result['half_lag_z'])
        result['full_corr'] = z.rolling(params.correlation_window, min_periods=params.correlation_window).corr(result['full_lag_z'])
        result = result.replace([np.inf, -np.inf], np.nan)
        result['entry'] = (
            (close > result['regime_ma'])
            & (result['atr'] > 0)
            & (result['half_corr'] <= -params.correlation_floor)
            & (result['full_corr'] >= params.correlation_floor)
            & (result['z'].shift(1) < -params.entry_z)
            & (result['z'] >= -params.entry_z)
            & (result['z'] < 0)
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CycleParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = indicators['regime_ma'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        entry_bar = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i]) and atr[i] > 0
            if active:
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.atr_multiple * atr[i])
                    exit_now = close[i] <= stop or close[i] <= regime[i] or i - entry_bar + 1 >= params.holding_bars
                else:
                    exit_now = True
                if exit_now:
                    active = False
                    continue
            elif valid and close[i] > regime[i] and entries[i]:
                active = True
                entry_bar = i + 1
                # Initialize the high-water mark on the first actual holding bar.
                high_water = -np.inf
                stop = -np.inf
            desired[i] = int(active)
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = CycleParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=225, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    actual = strategy.indicators(data, params)
    assert not actual['entry'].any()
    assert strategy.warmup_bars(params) >= 200
    controlled = pd.DataFrame({'atr': 1.0, 'regime_ma': 90.0, 'entry': False}, index=index)
    controlled.loc[index[200], 'entry'] = True
    signal = strategy.generate_signals(data, controlled, None, params).data['signal']
    assert signal.iloc[200] == 0 and signal.iloc[201:219].eq(1).all() and signal.iloc[219] == 0
    data.loc[index[202], 'close'] = 110.0
    data.loc[index[203], 'close'] = 104.0
    controlled.loc[index[203], 'atr'] = 10.0
    signal = strategy.generate_signals(data, controlled, None, params).data['signal']
    assert signal.iloc[203] == 1 and signal.iloc[204] == 0
    assert signal.isin([0, 1]).all()


if __name__ == '__main__':
    _self_check()
