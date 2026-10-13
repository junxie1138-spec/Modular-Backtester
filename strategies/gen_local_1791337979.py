from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    history_bars: int = 126
    complexity_edge: float = 0.05
    probability_edge: float = 0.04
    atr_period: int = 14
    initial_stop_atr: float = 2.0
    breakeven_gain: float = 0.01
    trail_atr: float = 2.5
    max_hold: int = 8
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791337979'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.history_bars + 5, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.history_bars < 30 or params.atr_period < 1:
            raise ValueError('history_bars must be >= 30 and atr_period >= 1')
        out = pd.DataFrame(index=data.index)
        r = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = r.shift(2), r.shift(1), r
        valid = a.notna() & b.notna() & c.notna()
        # Equal returns are ordered by their chronological positions.
        code = (a <= b).astype(int) + 2 * (a <= c).astype(int) + 4 * (b <= c).astype(int)
        out['return'] = r
        out['ordinal_pattern'] = code.where(valid)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out['weekday'] = weekday
        counts = {}
        for day in range(5):
            counts[day] = (weekday == day).astype(float).where(valid).rolling(params.history_bars).sum().shift(1)
            out[f'weekday_count_{day}'] = counts[day]
        base_entropy = pd.Series(0.0, index=data.index)
        seasonal_entropy = pd.Series(0.0, index=data.index)
        base_match = pd.Series(0.0, index=data.index)
        seasonal_match = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 3, 4, 6, 7):
            hit = (out['ordinal_pattern'] == pattern).astype(float).where(valid)
            base = hit.rolling(params.history_bars).mean().shift(1)
            seasonal = pd.Series(np.nan, index=data.index)
            for day in range(5):
                numerator = (hit * (weekday == day)).rolling(params.history_bars).sum().shift(1)
                probability = numerator / counts[day].replace(0.0, np.nan)
                seasonal = seasonal.where(weekday != day, probability)
            out[f'base_probability_{pattern}'] = base
            out[f'seasonal_probability_{pattern}'] = seasonal
            base_entropy -= base * np.log(base.clip(lower=1e-12))
            seasonal_entropy -= seasonal * np.log(seasonal.clip(lower=1e-12))
            base_match += base.where(out['ordinal_pattern'] == pattern, 0.0)
            seasonal_match += seasonal.where(out['ordinal_pattern'] == pattern, 0.0)
        out['base_complexity'] = base_entropy / np.log(6.0)
        out['seasonal_complexity'] = seasonal_entropy / np.log(6.0)
        out['complexity_advantage'] = out['base_complexity'] - out['seasonal_complexity']
        out['base_pattern_probability'] = base_match.where(valid)
        out['seasonal_pattern_probability'] = seasonal_match.where(valid)
        out['probability_advantage'] = out['seasonal_pattern_probability'] - out['base_pattern_probability']
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_period).mean()
        out['eligible'] = (
            (out['complexity_advantage'] >= params.complexity_edge)
            & (out['probability_advantage'] >= params.probability_edge)
            & (r > 0.0)
            & (out['atr'] > 0.0)
        ).fillna(False)
        out['entry_ready'] = out['eligible'] & out['eligible'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        if params.max_hold < 1 or params.size <= 0.0:
            raise ValueError('max_hold and size must be positive')
        if params.initial_stop_atr <= 0.0 or params.trail_atr <= 0.0 or params.breakeven_gain <= 0.0:
            raise ValueError('Stop distances and breakeven_gain must be positive')
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        held = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if active:
                if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0.0:
                    active = pending = False
                    continue
                if pending:
                    # The prior close's entry decision fills at this bar's open.
                    entry = opening
                    peak = entry
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    pending = False
                held += 1
                # Only test the stop established before this bar's price action.
                # A breach requests liquidation at the following open.
                if low <= stop or held >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                    if close <= stop:
                        active = False
                        continue
                signal[i] = 1
            elif ready[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(close) and close > 0.0:
                active = pending = True
                armed = False
                held = 0
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Checks next-open entry, breakeven activation, and next-open liquidation.
    index = pd.date_range('2020-01-06', periods=5, freq='B')
    data = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [101.0, 101.0, 102.0, 100.0, 100.0],
        'low': [99.5, 99.5, 100.5, 99.0, 98.0],
        'close': [100.0, 100.0, 101.5, 99.5, 99.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 5,
        'entry_ready': [False, True, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    prefix = strategy.generate_signals(data.iloc[:3], indicators.iloc[:3], ctx, params).data
    assert prefix.equals(result.iloc[:3])
