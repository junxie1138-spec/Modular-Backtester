from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    pattern_window: int = 24
    atr_window: int = 14
    complexity_min: float = 0.025
    rarity_max: float = 0.15
    gap_atr_min: float = 0.25
    breakeven_pct: float = 0.012
    trail_atr: float = 1.8
    hold_days: int = 4

    def __post_init__(self):
        if self.pattern_window < 3 or self.atr_window < 2:
            raise ValueError('Indicator windows are too short')
        if not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be between 3 and 5')
        if not 0 < self.complexity_min < 1 or not 0 < self.rarity_max < 1:
            raise ValueError('Complexity and rarity thresholds must lie in (0, 1)')
        if not all(np.isfinite(v) and v > 0 for v in (
            self.gap_atr_min, self.breakeven_pct, self.trail_atr
        )):
            raise ValueError('Gap and stop parameters must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = 'gen_local_1791240436'

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + 4, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['gap'] = data['open'].div(previous_close.where(previous_close > 0)).sub(1)
        out['gap'] = out['gap'].replace([np.inf, -np.inf], np.nan)
        a, b, c = out['gap'].shift(2), out['gap'].shift(1), out['gap']
        out['pattern'] = np.select([
            (a < b) & (b < c), (a < c) & (c < b),
            (b < a) & (a < c), (b < c) & (c < a),
            (c < a) & (a < b), (c < b) & (b < a)
        ], [0, 1, 2, 3, 4, 5], default=-1)
        valid_count = out['pattern'].ge(0).astype(float).rolling(params.pattern_window).sum()
        entropy = pd.Series(0.0, index=data.index)
        disequilibrium = pd.Series(0.0, index=data.index)
        for code in range(6):
            probability = out['pattern'].eq(code).astype(float).rolling(params.pattern_window).sum().div(valid_count.where(valid_count > 0))
            out[f'p{code}'] = probability
            entropy -= probability * np.log(probability.clip(lower=1e-12))
            disequilibrium += (probability - 1.0 / 6.0).pow(2)
        out['complexity'] = (entropy / np.log(6.0) * disequilibrium / (5.0 / 6.0)).where(valid_count >= params.pattern_window * 0.8)
        out['descending_rarity'] = out['p5'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_window).mean()
        out['gap_atr'] = (previous_close - data['open']).div(out['atr'].where(out['atr'] > 0))
        gap_depth = previous_close - data['open']
        out['repair'] = (data['close'] - data['open']).div(gap_depth.where(gap_depth > 0))
        out['ready'] = (
            out['pattern'].eq(5)
            & out['gap'].lt(0)
            & out['complexity'].ge(params.complexity_min)
            & out['descending_rarity'].le(params.rarity_max)
            & out['gap_atr'].ge(params.gap_atr_min)
            & out['repair'].ge(1.0)
        ).fillna(False)
        strength = (
            out['complexity'].div(3.0 * params.complexity_min).clip(0, 1)
            * (1.0 - out['descending_rarity']).clip(0, 1)
            * out['gap_atr'].div(2.0 * params.gap_atr_min).clip(0, 1)
        )
        out['conviction_size'] = (0.20 + 0.75 * strength).fillna(0.20).clip(0.20, 0.95)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({'signal': 0, 'size': 0.20}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction_size'].to_numpy(dtype=float)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        held_size = 0.20
        warmup = self.warmup_bars(params)
        for i, (opening, high, low, close) in enumerate(prices):
            valid_bar = np.isfinite([opening, high, low, close]).all() and min(opening, high, low, close) > 0
            if pending:
                pending = False
                if not valid_bar or i == 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                active = True
                armed = False
                entry = opening
                peak = opening
                stop = max(0.0, entry - params.trail_atr * atr[i - 1])
                age = 0
            if active:
                sizes[i] = held_size
                age += 1
                if not valid_bar or low <= stop or age >= params.hold_days:
                    active = False
                    continue
                peak = max(peak, high)
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                    # ponytail: new high-based stops are checked against the close;
                    # use intraday data if same-bar high/low ordering is required.
                    if close <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif i >= warmup - 1 and valid_bar and ready[i] and np.isfinite(atr[i]) and atr[i] > 0:
                held_size = float(np.clip(conviction[i], 0.20, 0.95)) if np.isfinite(conviction[i]) else 0.20
                pending = True
                signals[i] = 1
                sizes[i] = held_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapParams(pattern_window=3, atr_window=2)
    n = 14
    index = pd.date_range('2020-01-01', periods=n, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 100.5, 'low': 99.5, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'ready': False, 'conviction_size': 0.80}, index=index)
    decision = GeneratedStrategy.warmup_bars(None, params) - 1
    indicators.iloc[decision, indicators.columns.get_loc('ready')] = True
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    expected = np.zeros(n, dtype=int)
    expected[decision + 1:decision + 1 + params.hold_days] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert result['size'].gt(0).all()
    assert result['size'].iloc[decision + 1] == 0.80
    data.loc[index[decision + 1], ['high', 'close']] = [103.0, 102.0]
    data.loc[index[decision + 2], ['high', 'low', 'close']] = [102.0, 100.5, 101.0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    expected[:] = 0
    expected[decision + 1:decision + 3] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived.index.equals(data.index)
    assert not derived['ready'].any()


if __name__ == '__main__':
    _self_check()
