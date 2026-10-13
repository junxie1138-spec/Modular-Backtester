from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RetentionParams:
    ma_bars: int = 40
    slope_bars: int = 5
    atr_bars: int = 14
    entry_z: float = 0.8
    ceiling_z: float = 2.2
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.ma_bars, self.slope_bars, self.atr_bars, self.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not all(np.isfinite(v) for v in (self.entry_z, self.ceiling_z, self.stop_atr)):
            raise ValueError('Thresholds must be finite.')
        if not 0 < self.entry_z < self.ceiling_z or self.stop_atr <= 0:
            raise ValueError('Require 0 < entry_z < ceiling_z and stop_atr > 0.')


class GeneratedStrategy(BaseStrategy[RetentionParams]):
    strategy_id = 'gen_local-w2_1791288830'

    @classmethod
    def params_type(cls):
        return RetentionParams

    def warmup_bars(self, params):
        return int(max(params.ma_bars + params.slope_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out['dispersion'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out['z'] = (close - out['ma']) / out['dispersion'].where(out['dispersion'] > 0)
        out['ma_change'] = out['ma'].diff(params.slope_bars)
        out['body'] = close - data['open']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['long_bar'] = (
            out['z'].between(params.entry_z, params.ceiling_z)
            & (out['ma_change'] > 0) & (out['body'] < 0)
        )
        out['short_bar'] = (
            out['z'].between(-params.ceiling_z, -params.entry_z)
            & (out['ma_change'] < 0) & (out['body'] > 0)
        )
        out['confirmed'] = np.select([
            out['long_bar'] & out['long_bar'].shift(1, fill_value=False),
            out['short_bar'] & out['short_bar'].shift(1, fill_value=False),
        ], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['confirmed'].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        pending_entry = False
        entry, frozen_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                if pending_entry:
                    # The prior close's decision fills at this bar's open.
                    entry = opens[i]
                    pending_entry = False
                age += 1
                stopped = position * (close[i] - entry) < -params.stop_atr * frozen_atr
                if not np.isfinite(entry) or not np.isfinite(close[i]) or stopped or age >= params.hold_bars:
                    position = 0
                raw[i] = position
                continue
            if candidates[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(candidates[i])
                frozen_atr = atr[i]
                age = 0
                pending_entry = True
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    for direction in (1, -1):
        entry = 100.0 + 10.0 * direction
        data = pd.DataFrame({
            'open': [100.0, 100.0] + [entry] * 5,
            'close': [100.0, 100.0, entry - direction, entry - 3 * direction, entry, entry, entry],
            'high': [130.0] * 7, 'low': [70.0] * 7, 'volume': [1000.0] * 7,
        }, index=index)
        ind = pd.DataFrame({
            'confirmed': [0, direction, 0, 0, 0, 0, 0],
            'atr': [1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, RetentionParams()).data
        assert result['signal'].tolist() == [0, 0, direction, direction, 0, 0, 0]
        assert result['size'].gt(0).all()
        data['close'] = entry
        result = GeneratedStrategy.generate_signals(None, data, ind, None, RetentionParams(hold_bars=2)).data
        assert result['signal'].tolist() == [0, 0, direction, direction, 0, 0, 0]
    closes = np.arange(90.0, 110.0, 2.0)
    trend = pd.DataFrame({'open': closes + 1, 'high': closes + 2,
                          'low': closes - 2, 'close': closes, 'volume': 1000.0},
                         index=pd.date_range('2020-01-01', periods=len(closes)))
    params = RetentionParams(ma_bars=3, slope_bars=1, atr_bars=2)
    ind = GeneratedStrategy.indicators(None, trend, params)
    assert ind['long_bar'].iloc[3] and not ind['confirmed'].iloc[3]
    assert ind['confirmed'].iloc[4] == 1
    result = GeneratedStrategy.generate_signals(None, trend, ind, None, params).data
    assert result['signal'].iloc[:5].eq(0).all() and result['signal'].iloc[5] == 1
    flat = pd.DataFrame({c: 100.0 for c in ('open', 'high', 'low', 'close', 'volume')},
                        index=pd.date_range('2020-01-01', periods=80))
    ind = GeneratedStrategy.indicators(None, flat, RetentionParams())
    assert ind['confirmed'].eq(0).all()
    result = GeneratedStrategy.generate_signals(None, flat, ind, None, RetentionParams()).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
