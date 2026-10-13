from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarStretchParams:
    ma_window: int = 20
    seasonal_window: int = 252
    tail_percentile: float = 0.20
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[CalendarStretchParams]):
    strategy_id = 'gen_local_1791209221'

    @classmethod
    def params_type(cls):
        return CalendarStretchParams

    def warmup_bars(self, params: CalendarStretchParams) -> int:
        return int(max(params.ma_window + params.seasonal_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CalendarStretchParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.seasonal_window < 100 or params.atr_window < 1:
            raise ValueError('Invalid lookback parameters')
        if not 0.0 < params.tail_percentile < 0.5:
            raise ValueError('tail_percentile must be between zero and one half')
        if params.breakeven_pct <= 0.0 or params.trail_atr <= 0.0:
            raise ValueError('Stop parameters must be positive')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['weekday'] = data.index.dayofweek
        out['lower'] = np.nan
        out['upper'] = np.nan
        for weekday in range(5):
            mask = out['weekday'].eq(weekday)
            history = out['z'].where(mask).shift(1).rolling(params.seasonal_window, min_periods=20)
            lower = history.quantile(params.tail_percentile)
            upper = history.quantile(1.0 - params.tail_percentile)
            out.loc[mask, 'lower'] = lower.loc[mask]
            out.loc[mask, 'upper'] = upper.loc[mask]
        ready = out[['z', 'lower', 'upper', 'atr']].notna().all(axis=1) & out['atr'].gt(0.0)
        long_entry = ready & out['weekday'].le(1) & out['z'].lt(0.0) & out['z'].lt(out['lower'])
        short_entry = ready & out['weekday'].ge(3) & out['weekday'].le(4) & out['z'].gt(0.0) & out['z'].gt(out['upper'])
        out['entry_signal'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CalendarStretchParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry_signal'].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        side = age = 0
        entry_bar = -1
        entry = stop = 0.0
        armed = False
        for i in range(len(data)):
            if side == 0:
                if candidates[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                    side = int(candidates[i])
                    entry_bar = i + 1
                    raw[i] = side
                continue
            if not np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all() or opens[i] <= 0.0 or atr[i] <= 0.0:
                side = 0
                continue
            if i == entry_bar:
                entry = side * opens[i]
                stop = entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
            age += 1
            # Signed prices make the stop increase monotonically for both directions.
            adverse = lows[i] if side == 1 else -highs[i]
            price = side * closes[i]
            # A touched stop schedules liquidation at the next open under this contract.
            # Test the existing stop before updating it; OHLC does not reveal intrabar order.
            if adverse <= stop:
                side = 0
                continue
            if price - entry >= abs(entry) * params.breakeven_pct:
                armed = True
            if armed:
                stop = max(stop, entry, price - params.trail_atr * atr[i])
            if age >= 2:
                side = 0
            raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = CalendarStretchParams()
    index = pd.date_range('2020-01-06', periods=5, freq='B')
    for direction in (1, -1):
        opens = np.full(5, 100.0)
        closes = opens.copy()
        closes[1] += direction * 0.6
        data = pd.DataFrame({'open': opens, 'high': np.maximum(opens, closes) + 0.1, 'low': np.minimum(opens, closes) - 0.1, 'close': closes, 'volume': np.ones(5)}, index=index)
        indicators = pd.DataFrame({'atr': np.ones(5), 'entry_signal': [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]
        assert result['size'].gt(0.0).all()
        # With no stop touch, expiry still occurs after two occupied sessions.
        data['high'] = np.maximum(opens, closes) + 0.01
        data['low'] = np.minimum(opens, closes) - 0.01
        data.loc[index[2], ['open', 'high', 'low', 'close']] = [100.0 + direction * 0.7, 100.8 if direction == 1 else 99.4, 100.6 if direction == 1 else 99.2, 100.0 + direction * 0.7]
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]


if __name__ == '__main__':
    _self_check()
