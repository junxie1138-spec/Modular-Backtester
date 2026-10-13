from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    history: int = 126
    atr_period: int = 14
    min_analogs: int = 5
    forecast_threshold: float = 0.001
    breakeven_pct: float = 0.005
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = 'gen_local-w2_1791322362'

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    def warmup_bars(self, params):
        return int(max(params.history + 8, params.atr_period + 1))

    def indicators(self, data, params):
        if params.history < 3 or params.atr_period < 1 or params.min_analogs < 1:
            raise ValueError('Lookbacks and minimum analog count must be positive.')
        if not np.isfinite(params.forecast_threshold):
            raise ValueError('Forecast threshold must be finite.')
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError('Breakeven activation must be positive and finite.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('ATR multiplier must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        change = close.diff()
        up = change.gt(0)
        down = change.lt(0)
        streak = pd.Series(0, index=data.index, dtype=int)
        up_run = pd.Series(True, index=data.index)
        down_run = up_run.copy()
        for offset in range(4):
            up_run &= up.shift(offset, fill_value=False)
            down_run &= down.shift(offset, fill_value=False)
            streak += up_run.astype(int) - down_run.astype(int)
        out['streak'] = streak
        out['streak_previous'] = streak.shift(1)
        out['streak_earlier'] = streak.shift(2)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period, min_periods=params.atr_period).mean()
        # At bar j+2 this is the fully realized outcome of shape j.
        out['matured_return'] = (close / data['open'].shift(1).where(data['open'].shift(1).gt(0)) - 1).replace([np.inf, -np.inf], np.nan)
        counts = pd.Series(0, index=data.index, dtype=int)
        totals = pd.Series(0.0, index=data.index)
        shape_columns = ['streak', 'streak_previous', 'streak_earlier']
        # Historical endpoints precede the current bar by at least three bars.
        for lag in range(3, params.history + 3):
            matched = pd.Series(True, index=data.index)
            for column in shape_columns:
                matched &= out[column].eq(out[column].shift(lag))
            outcome = out['matured_return'].shift(lag - 2)
            matched &= outcome.notna()
            matched &= pd.Series(np.arange(len(data)) >= lag + 4, index=data.index)
            counts += matched.astype(int)
            totals += outcome.where(matched, 0.0)
        out['analog_count'] = counts
        out['forecast'] = totals.div(counts.where(counts.gt(0)))
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params) - 1, index=data.index)
        out['candidate'] = (
            ready & out['atr'].gt(0) & out['streak'].gt(0)
            & counts.ge(params.min_analogs)
            & out['forecast'].gt(params.forecast_threshold)
        ).fillna(False)
        out['confirmed'] = out['candidate'] & out['candidate'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        n = len(data)
        target = np.zeros(n, dtype=int)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        active = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_bar = -1
        for i in range(n):
            opening, high, low, close = prices[i]
            valid = bool(np.isfinite(prices[i]).all() and (prices[i] > 0).all())
            # The previous close's target enters at this bar's open.
            if not active and i > 0 and target[i - 1] == 1:
                if not valid or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                active = True
                armed = False
                entry = opening
                entry_bar = i
                stop = max(0.0, entry - params.trail_atr * atr[i - 1])
            if active:
                if not valid or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # An observed stop breach requests an exit at the next open.
                breached = low <= stop
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, high - params.trail_atr * atr[i])
                # Newly raised stops use the close, avoiding assumptions about
                # the intrabar ordering of today's high and low.
                breached = breached or close <= stop
                if breached or i - entry_bar + 1 >= 2:
                    active = False
                else:
                    target[i] = 1
            elif valid and confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                target[i] = 1
        df = pd.DataFrame({'signal': target, 'size': np.ones(n, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakAnalogParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [100.4, 100.4, 102.0, 100.4, 100.4, 100.4],
        'low': [99.9] * 6,
        'close': [100.2] * 6,
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 6,
        'confirmed': [False, True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0]
    data.loc[index[2], 'high'] = 100.4
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    derived = strategy.indicators(flat, params)
    assert not derived['confirmed'].any()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
