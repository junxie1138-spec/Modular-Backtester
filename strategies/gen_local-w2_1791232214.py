from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    entropy_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.25
    momentum_window: int = 3
    atr_window: int = 14
    trail_atr: float = 1.5
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.entropy_window, self.percentile_window) < 2:
            raise ValueError('Entropy and percentile windows must be at least two.')
        if min(self.momentum_window, self.atr_window, self.holding_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('Entry percentile must lie between zero and one.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('ATR multiplier must be finite and positive.')


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local-w2_1791232214'

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(
            params.entropy_window + params.percentile_window + 4,
            params.momentum_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['daily_return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag_1'] = out['daily_return'].shift(1)
        out['return_lag_2'] = out['daily_return'].shift(2)
        triples = out[['return_lag_2', 'return_lag_1', 'daily_return']].to_numpy()
        order = np.argsort(triples, axis=1, kind='stable')
        # Stable sorting breaks equal-return ties in chronological order.
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        valid = np.isfinite(triples).all(axis=1)
        out['pattern'] = pd.Series(codes, index=data.index, dtype=float).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = out['pattern'].eq(code).astype(float).where(valid)
            probability = occurrence.rolling(params.entropy_window, min_periods=params.entropy_window).mean()
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.where(probability > 0, 1.0))
        out['entropy'] = entropy / np.log(6.0)
        out['entropy_threshold'] = out['entropy'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile)
        out['momentum'] = close.pct_change(params.momentum_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1).where(
            data[['high', 'low', 'close']].notna().all(axis=1) & previous_close.notna()
        )
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        contraction = (
            out['entropy'].le(out['entropy_threshold'])
            & out['entropy'].shift(1).gt(out['entropy_threshold'].shift(1))
        )
        ready = np.isfinite(out[['entropy', 'entropy_threshold', 'momentum', 'daily_return', 'atr']]).all(axis=1)
        aligned = out['momentum'] * out['daily_return'] > 0
        out['entry'] = np.where(
            contraction & ready & aligned & out['atr'].gt(0),
            np.sign(out['momentum'].fillna(0)),
            0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = np.nan
        lowest = np.nan
        stop = np.nan
        distance = np.nan
        for i, close in enumerate(closes):
            if not np.isfinite(close) or close <= 0:
                position = 0
                continue
            if position:
                # Entry decisions fill at the next open; extrema start at that day's close.
                age += 1
                highest = close if np.isnan(highest) else max(highest, close)
                lowest = close if np.isnan(lowest) else min(lowest, close)
                if position == 1:
                    candidate = highest - distance
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = close <= stop
                else:
                    # Mirror the rolling-high trail for short positions.
                    candidate = lowest + distance
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.holding_bars:
                    position = 0
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                age = 0
                highest = lowest = stop = np.nan
                # Fix the ATR distance at the decision close so the stop never loosens.
                distance = params.trail_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = OrdinalParams()
    index = pd.date_range('2020-01-01', periods=6)
    for direction, prices in (
        (1, [100, 102, 104, 101, 100, 100]),
        (-1, [100, 98, 96, 99, 100, 100]),
    ):
        data = pd.DataFrame({'close': prices}, index=index)
        derived = pd.DataFrame({'atr': 1.0, 'entry': [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0, 0]
        assert result['size'].gt(0).all()
    data = pd.DataFrame({'close': 100.0}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'entry': [1, 0, 0, 0, 0, 0]}, index=index)
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
