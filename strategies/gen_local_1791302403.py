from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 12
    range_window: int = 14
    atr_window: int = 14
    complexity_ceiling: float = 0.67
    position_ceiling: float = 0.45
    breakeven_pct: float = 0.006
    trail_atr: float = 0.9
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791302403'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if min(params.pattern_window, params.range_window, params.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0 < params.complexity_ceiling <= 1:
            raise ValueError('Complexity ceiling must be in (0, 1].')
        if not 0 < params.position_ceiling <= 1:
            raise ValueError('Position ceiling must be in (0, 1].')
        if not params.breakeven_pct > 0 or not params.trail_atr > 0:
            raise ValueError('Exit parameters must be positive.')
        if params.hold_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions.')
        return int(max(params.pattern_window + 3, params.range_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        valid &= (a != b) & (a != c) & (b != c)
        pattern = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).astype(float)
        out['pattern'] = pattern.where(valid)
        codes = (0, 1, 3, 4, 6, 7)
        count_columns = []
        for code in codes:
            name = 'prior_count_' + str(code)
            occurrence = out['pattern'].eq(code).astype(float).where(valid)
            out[name] = occurrence.shift(1).rolling(params.pattern_window, min_periods=params.pattern_window).sum()
            count_columns.append(name)
        ready = out[count_columns].notna().all(axis=1)
        out['complexity'] = (out[count_columns].gt(0).sum(axis=1) / 6.0).where(ready)
        out['range_high'] = high.rolling(params.range_window, min_periods=params.range_window).max()
        out['range_low'] = low.rolling(params.range_window, min_periods=params.range_window).min()
        span = (out['range_high'] - out['range_low']).replace(0.0, np.nan)
        out['position'] = ((close - out['range_low']) / span).clip(0.0, 1.0)
        out['close_location'] = ((close - low) / (high - low).replace(0.0, np.nan)).clip(0.0, 1.0)
        out['true_range'] = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1).where(close.shift(1).notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['new_valley'] = ((out['pattern'].eq(4) & out['prior_count_4'].eq(0)) | (out['pattern'].eq(6) & out['prior_count_6'].eq(0)))
        out['strength'] = ((1.0 - out['complexity']) * (1.0 - out['position']) * out['close_location']).clip(0.0, 1.0).fillna(0.0)
        out['entry'] = (ready & out['new_valley'] & out['complexity'].le(params.complexity_ceiling) & out['position'].le(params.position_ceiling) & out['close_location'].ge(0.6) & out['atr'].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        active = False
        pending = False
        armed = False
        entry = 0.0
        peak = 0.0
        stop = -np.inf
        age = 0
        position_size = 0.2
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.0).to_numpy(dtype=float)
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite([opening, high, low, close, atrs[i]]).all()
            valid = valid and min(opening, high, low, close) > 0 and atrs[i] > 0
            valid = valid and low <= min(opening, close) and high >= max(opening, close)
            if not valid:
                active = False
                pending = False
                continue
            if pending:
                # Reference entry is the next session's open, known only now.
                active = True
                pending = False
                entry = opening
                peak = opening
                stop = -np.inf
                armed = False
                age = 0
            if active:
                age += 1
                # Check only the previously established stop against today's low.
                exit_now = armed and low <= stop
                peak = max(peak, high)
                if not exit_now:
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                        # A new stop is checked against the close, avoiding unknown intrabar order.
                        exit_now = close <= stop
                exit_now = exit_now or age >= params.hold_bars
                if exit_now:
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif entries[i]:
                strength = strengths[i] if np.isfinite(strengths[i]) else 0.0
                position_size = float(0.2 + 0.75 * np.clip(2.0 * strength, 0.0, 1.0))
                signals[i] = 1
                sizes[i] = position_size
                pending = True
        df['signal'] = signals
        df['size'] = sizes
        # Stops produce next-open exit orders, not assumed intraday stop fills.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': [100.2, 102.0, 100.2, 100.2, 100.2, 100.2, 100.2], 'low': 99.8, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'entry': [True, False, False, True, False, False, False], 'strength': [0.1, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert 0 < result['size'].iloc[1] < result['size'].iloc[4] <= 0.95
    assert result['size'].gt(0).all()
    assert strategy.indicators(data, params)['entry'].eq(False).all()
