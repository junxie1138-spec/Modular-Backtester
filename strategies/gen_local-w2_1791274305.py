from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    channel_bars: int = 20
    atr_bars: int = 14
    test_bars: int = 8
    min_tests: int = 3
    test_distance_atr: float = 0.35
    expansion_bars: int = 10
    expansion_ratio: float = 1.05
    breakout_buffer_atr: float = 0.15
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold_bars: int = 10

    def __post_init__(self):
        windows = (self.channel_bars, self.atr_bars, self.test_bars,
                   self.expansion_bars, self.max_hold_bars)
        if any(v < 1 for v in windows):
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 1 <= self.min_tests <= self.test_bars:
            raise ValueError('min_tests must be within test_bars.')
        values = (self.test_distance_atr, self.expansion_ratio,
                  self.breakout_buffer_atr, self.breakeven_pct, self.trail_atr)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Thresholds must be finite and positive.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791274305'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(200,
                       max(params.channel_bars, params.atr_bars + 1)
                       + params.test_bars + 1,
                       params.atr_bars + params.expansion_bars + 1))

    def indicators(self, data, params):
        p = params
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_bars).mean()
        out['ma200'] = data['close'].rolling(200).mean()
        out['ceiling'] = data['high'].rolling(p.channel_bars).max()
        out['resistance'] = out['ceiling'].shift(1)
        out['atr_baseline'] = out['atr'].shift(1).rolling(p.expansion_bars).mean()
        ready = out['ceiling'].notna() & out['atr'].gt(0)
        out['resistance_test'] = (
            ready
            & data['high'].ge(out['ceiling'] - p.test_distance_atr * out['atr'])
            & data['close'].lt(out['ceiling'])
        ).astype(float).where(ready)
        out['prior_tests'] = out['resistance_test'].shift(1).rolling(p.test_bars).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            data['close'].gt(out['ma200'])
            & data['close'].gt(out['resistance'] + p.breakout_buffer_atr * out['atr'])
            & out['prior_tests'].ge(p.min_tests)
            & out['atr'].gt(p.expansion_ratio * out['atr_baseline'])
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry_price = peak = stop = entry_atr = 0.0
        age = 0

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    peak = entry_price
                    stop = entry_price - p.trail_atr * entry_atr
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                # Close-confirmed exits execute through the next-bar signal.
                if close <= stop or age >= p.max_hold_bars:
                    active = False
                    continue
                peak = max(peak, close)
                if close >= entry_price * (1.0 + p.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - p.trail_atr * atr)
                    if close <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif valid and entries[i]:
                signals[i] = 1
                pending = True
                entry_atr = atr

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    closes = np.array([100.0, 100.0, 103.0, 104.0, 100.5, 100.0, 100.0])
    data = pd.DataFrame({'open': 100.0, 'high': closes + 1.0,
                         'low': closes - 1.0, 'close': closes,
                         'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0],
                            'entry': [True, False, False, False, False, False, False]},
                           index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, StrategyParams()).data
    # ATR expansion must not lower the stop established at 101 on day three.
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    expiry = strategy.generate_signals(
        data, derived, None, StrategyParams(max_hold_bars=2)).data
    assert expiry['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    warmup = strategy.indicators(data, StrategyParams())
    assert not warmup['entry'].any()
    assert strategy.warmup_bars(StrategyParams()) >= 200


if __name__ == '__main__':
    _self_check()
