from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    streak_bars: int = 3
    displacement_ratio: float = 1.5
    hold_bars: int = 7

    def __post_init__(self):
        if not isinstance(self.streak_bars, int) or not 2 <= self.streak_bars <= 9:
            raise ValueError('streak_bars must be an integer between 2 and 9')
        if not np.isfinite(self.displacement_ratio) or self.displacement_ratio <= 0:
            raise ValueError('displacement_ratio must be finite and positive')
        if not isinstance(self.hold_bars, int) or not 5 <= self.hold_bars <= 10:
            raise ValueError('hold_bars must be an integer between 5 and 10')


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = 'gen_local_1791220799'

    @classmethod
    def params_type(cls):
        return MigrationParams

    def warmup_bars(self, params: MigrationParams) -> int:
        return params.streak_bars + 1

    def indicators(self, data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        n = params.streak_bars
        out = pd.DataFrame(index=data.index)
        change = data['close'].diff()
        low_change = data['low'].diff()
        high_change = data['high'].diff()
        up = (change > 0) & (low_change > 0) & (data['close'] > data['open'])
        down = (change < 0) & (high_change < 0) & (data['close'] < data['open'])
        up_run = up.copy()
        down_run = down.copy()
        up_count = up.astype(int)
        down_count = down.astype(int)
        for lag in range(1, n):
            up_run = up_run & up.shift(lag, fill_value=False)
            down_run = down_run & down.shift(lag, fill_value=False)
            up_count = up_count + up_run.astype(int)
            down_count = down_count + down_run.astype(int)
        body_low = data[['open', 'close']].min(axis=1, skipna=False)
        body_high = data[['open', 'close']].max(axis=1, skipna=False)
        lower_wick = (body_low - data['low']).clip(lower=0)
        upper_wick = (data['high'] - body_high).clip(lower=0)
        out['close_change'] = change
        out['low_change'] = low_change
        out['high_change'] = high_change
        out['up_step'] = up.astype(int)
        out['down_step'] = down.astype(int)
        out['up_streak'] = up_count
        out['down_streak'] = down_count
        out['body_low'] = body_low
        out['body_high'] = body_high
        out['lower_wick'] = lower_wick
        out['upper_wick'] = upper_wick
        out['displacement'] = change.rolling(n, min_periods=n).sum()
        out['lower_wick_sum'] = lower_wick.rolling(n, min_periods=n).sum()
        out['upper_wick_sum'] = upper_wick.rolling(n, min_periods=n).sum()
        out['ready'] = out[['displacement', 'lower_wick_sum', 'upper_wick_sum']].notna().all(axis=1).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        displacement = indicators['displacement']
        long_entry = (
            (indicators['ready'] == 1)
            & (indicators['up_streak'] == params.streak_bars)
            & (displacement > 0)
            & (displacement >= params.displacement_ratio * indicators['lower_wick_sum'])
        )
        short_entry = (
            (indicators['ready'] == 1)
            & (indicators['down_streak'] == params.streak_bars)
            & (displacement < 0)
            & (-displacement >= params.displacement_ratio * indicators['upper_wick_sum'])
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = MigrationParams()
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        close = 100.0 + direction * np.arange(24, dtype=float)
        opening = close - direction * 0.3
        data = pd.DataFrame({
            'open': opening,
            'high': np.maximum(opening, close) + 0.1,
            'low': np.minimum(opening, close) - 0.1,
            'close': close,
            'volume': 1000000.0,
        }, index=pd.date_range('2020-01-01', periods=24, freq='B'))
        derived = strategy.indicators(data, params)
        result = strategy.generate_signals(data, derived, None, params).data
        entry = params.streak_bars + 1
        assert derived.index.equals(data.index)
        assert not derived.isna().any().any()
        assert (result['signal'].iloc[:entry] == 0).all()
        assert (result['signal'].iloc[entry:entry + params.hold_bars] == direction).all()
        assert result['signal'].iloc[entry + params.hold_bars] == 0
        assert set(result['signal'].unique()) <= {-1, 0, 1}
        assert (result['size'] > 0).all()
        assert strategy.warmup_bars(params) <= 10


if __name__ == '__main__':
    _self_check()
