from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelayParams:
    fast_days: int = 3
    medium_days: int = 6
    relay_days: int = 3
    atr_days: int = 14
    acceleration_threshold: float = 0.25
    stop_atr: float = 2.0
    hold_days: int = 8

    def __post_init__(self):
        if min(self.fast_days, self.relay_days, self.atr_days, self.hold_days) < 1:
            raise ValueError('Lookbacks and holding period must be positive')
        if self.medium_days <= self.fast_days:
            raise ValueError('medium_days must exceed fast_days')
        if not (np.isfinite(self.acceleration_threshold) and self.acceleration_threshold > 0):
            raise ValueError('acceleration_threshold must be finite and positive')
        if not (np.isfinite(self.stop_atr) and self.stop_atr > 0):
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[RelayParams]):
    strategy_id = 'gen_local-w2_1791248422'

    @classmethod
    def params_type(cls):
        return RelayParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.atr_days + 1,
                       2 * params.medium_days + 2,
                       2 * params.fast_days + params.relay_days + 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_days, min_periods=params.atr_days).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['atr_fraction'] = out['atr'] / close.where(close > 0)
        scale = out['atr_fraction'].where(out['atr_fraction'] > 0)
        for name, days in [('fast', params.fast_days), ('medium', params.medium_days)]:
            out[name + '_roc'] = close.pct_change(days, fill_method=None)
            out[name + '_acceleration'] = (
                out[name + '_roc'] - out[name + '_roc'].shift(days)
            ) / (days * scale)
        out['prior_fast_max'] = out['fast_acceleration'].shift(1).rolling(
            params.relay_days, min_periods=params.relay_days
        ).max()
        out['prior_fast_min'] = out['fast_acceleration'].shift(1).rolling(
            params.relay_days, min_periods=params.relay_days
        ).min()
        out['previous_medium_acceleration'] = out['medium_acceleration'].shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out.notna().all(axis=1) & out['atr'].gt(0)
        threshold = params.acceleration_threshold
        long_entry = (
            out['ready'] & close.gt(out['ma200'])
            & out['fast_roc'].gt(0) & out['medium_roc'].gt(0)
            & out['prior_fast_max'].gt(threshold)
            & out['medium_acceleration'].gt(threshold)
            & out['previous_medium_acceleration'].le(threshold)
        )
        short_entry = (
            out['ready'] & close.lt(out['ma200'])
            & out['fast_roc'].lt(0) & out['medium_roc'].lt(0)
            & out['prior_fast_min'].lt(-threshold)
            & out['medium_acceleration'].lt(-threshold)
            & out['previous_medium_acceleration'].ge(-threshold)
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        ready = indicators['ready'].to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending_fill = False
        fixed_stop = np.nan
        entry_atr = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                if pending_fill:
                    # The prior close's order fills at this bar's open.
                    entry_price = opens[i]
                    fixed_stop = entry_price - position * params.stop_atr * entry_atr
                    pending_fill = False
                age += 1
                stop_hit = position * (closes[i] - fixed_stop) < 0
                if (stop_hit or age >= params.hold_days
                        or not np.isfinite(closes[i]) or not np.isfinite(fixed_stop)):
                    position = 0
            elif ready[i] and entries[i] != 0:
                position = int(entries[i])
                entry_atr = atr[i]
                pending_fill = True
                age = 0
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [102, 112, 102, 102, 102, 107, 102],
        'low': [98, 98, 94, 98, 98, 98, 98],
        'close': [100, 110, 95, 100, 100, 105, 100],
        'volume': [1000.0] * 7
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2, 20, 20, 2, 2, 20, 2],
        'ready': [True] * 7,
        'entry': [1, 0, 0, 0, -1, 0, 0]
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, RelayParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, -1, 0]
    assert result.data['size'].gt(0).all()
