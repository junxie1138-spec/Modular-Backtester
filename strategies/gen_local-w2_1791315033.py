from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRepairParams:
    peak_window: int = 20
    percentile_window: int = 60
    atr_window: int = 14
    tail_quantile: float = 0.8
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    initial_stop_atr: float = 1.5
    max_hold: int = 2

    def __post_init__(self):
        if min(self.peak_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 0.5 < self.tail_quantile < 1.0:
            raise ValueError('tail_quantile must lie between 0.5 and 1.')
        if min(self.breakeven_pct, self.trail_atr, self.initial_stop_atr) <= 0:
            raise ValueError('Exit parameters must be positive.')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[RangeRepairParams]):
    strategy_id = 'gen_local-w2_1791315033'

    @classmethod
    def params_type(cls):
        return RangeRepairParams

    def warmup_bars(self, params):
        return int(max(params.peak_window, params.atr_window + 1, 3) + params.percentile_window + 1)

    def indicators(self, data, params):
        high, low, close = data['high'], data['low'], data['close']
        previous_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0, np.nan)
        peak = high.rolling(params.peak_window, min_periods=params.peak_window).max().replace(0, np.nan)
        drawdown = 1.0 - close / peak
        stress_threshold = drawdown.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_quantile).shift(1)
        overlap = (np.minimum(high, high.shift(1)) - np.maximum(low, low.shift(1))).clip(lower=0)
        previous_width = (high - low).shift(1).replace(0, np.nan)
        midpoint = (high + low) / 2.0
        restitution = overlap / previous_width * midpoint.diff() / atr
        upper = restitution.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_quantile).shift(1)
        lower = restitution.rolling(params.percentile_window, min_periods=params.percentile_window).quantile(1.0 - params.tail_quantile).shift(1)
        separated = high.shift(1) < low.shift(2)
        stressed = drawdown.shift(1) >= stress_threshold.shift(1)
        ready = pd.concat([atr, drawdown, stress_threshold, restitution, upper, lower], axis=1).notna().all(axis=1)
        eligible = ready & separated & stressed & (overlap > 0)
        long_entry = eligible & (restitution > upper) & (midpoint > midpoint.shift(1)) & (low >= low.shift(1))
        short_entry = eligible & (restitution < lower) & (midpoint < midpoint.shift(1)) & (high <= high.shift(1))
        entry = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        return pd.DataFrame({
            'true_range': true_range,
            'atr': atr,
            'peak': peak,
            'drawdown': drawdown,
            'stress_threshold': stress_threshold,
            'overlap': overlap,
            'previous_width': previous_width,
            'midpoint': midpoint,
            'restitution': restitution,
            'upper_threshold': upper,
            'lower_threshold': lower,
            'separated': separated,
            'stressed': stressed,
            'entry': entry,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        direction = pending = age = 0
        entry_price = best = stop = 0.0
        armed = False
        for i in range(len(data)):
            if direction == 0 and pending != 0:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    direction = pending
                    entry_price = opens[i]
                    best = direction * entry_price
                    stop = best - params.initial_stop_atr * atr[i - 1]
                    age = 0
                    armed = False
                pending = 0
            if direction != 0:
                age += 1
                valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
                exit_now = not valid
                if valid:
                    signed_close = direction * closes[i]
                    exit_now = signed_close <= stop
                    if not exit_now:
                        best = max(best, signed_close)
                        if best - direction * entry_price >= params.breakeven_pct * entry_price:
                            armed = True
                        if armed:
                            # Signed stop only rises; the corresponding short price stop only falls.
                            stop = max(stop, direction * entry_price, best - params.trail_atr * atr[i])
                            exit_now = signed_close <= stop
                if exit_now or age >= params.max_hold:
                    direction = 0
                raw[i] = direction
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(entries[i])
                raw[i] = pending
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        # All exits are close decisions executed at the following opening.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeRepairParams(trail_atr=0.25)
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({'open': [100.0] * 4, 'high': [102.0] * 4, 'low': [98.0] * 4, 'close': [100.0, 100.6, 100.0, 100.0], 'volume': [1000.0] * 4}, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 4, 'entry': [1, 0, 0, 0]}, index=index)
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].tolist() == [0, 1, 1, 0]
    data['close'] = [100.0, 98.0, 98.0, 98.0]
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].tolist() == [0, 1, 0, 0]
    indicators['entry'] = [-1, 0, 0, 0]
    data['close'] = [100.0, 102.0, 102.0, 102.0]
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].tolist() == [0, -1, 0, 0]
    data['close'] = [100.0, 100.0, 100.0, 100.0]
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].tolist() == [0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
