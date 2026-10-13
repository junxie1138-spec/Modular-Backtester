from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_days: int = 3
    volume_days: int = 20
    atr_days: int = 14
    stop_atr: float = 2.0
    holding_days: int = 4

    def __post_init__(self):
        if self.streak_days < 1 or self.volume_days < 1 or self.atr_days < 1:
            raise ValueError('Lookbacks must be positive.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')
        if self.holding_days < 3 or self.holding_days > 5:
            raise ValueError('holding_days must be between 3 and 5.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791266364'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(200, params.volume_days, params.atr_days + 1) + params.streak_days + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['price_change'] = data['close'].diff()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_days, min_periods=params.atr_days).mean()
        out['volume_mean'] = data['volume'].rolling(params.volume_days, min_periods=params.volume_days).mean()
        out['relative_volume'] = data['volume'].div(out['volume_mean'].where(out['volume_mean'] > 0))
        out['productivity'] = out['price_change'].div(out['atr'].where(out['atr'] > 0)).div(out['relative_volume'].where(out['relative_volume'] > 0))
        out['productivity'] = out['productivity'].replace([np.inf, -np.inf], np.nan)
        out['previous_productivity'] = out['productivity'].shift(1)
        out['strengthening'] = (
            out['productivity'].gt(0)
            & out['previous_productivity'].gt(0)
            & out['productivity'].gt(out['previous_productivity'])
        ).fillna(False)
        out['streak'] = out['strengthening'].astype(int).groupby((~out['strengthening']).cumsum()).cumsum()
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['entry'] = (
            out['streak'].eq(params.streak_days)
            & data['close'].gt(out['ma200'])
            & out['atr'].gt(0)
            & out['productivity'].notna()
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop = 0.0
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Entry is the next session's open; ATR is frozen at the decision close.
                    stop = opens[i] - params.stop_atr * entry_atr
                if not np.isfinite(close[i]) or close[i] < stop or i - entry_bar + 1 >= params.holding_days:
                    active = False
            elif entry[i] and np.isfinite(atr[i]) and atr[i] > 0 and i + 1 < len(data):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
            signal[i] = int(active)
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': 2.0, 'entry': [True, False, False, False, False, False]}, index=index)
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    data.loc[index[2], 'close'] = 95.0
    derived.loc[index[2]:, 'atr'] = 20.0
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    data['close'] = [100.0, 110.0, 104.0, 100.0, 100.0, 100.0]
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    data.loc[index[1], 'open'] = 110.0
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
