from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 63
    reference_window: int = 252
    tail_quantile: float = 0.8
    dwell_bars: int = 3
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791293771'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.peak_window + params.reference_window + params.dwell_bars + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.peak_window, p.reference_window, p.dwell_bars, p.atr_window, p.max_hold) < 1 or not 0 < p.tail_quantile < 1 or p.breakeven_pct <= 0 or p.trail_atr <= 0:
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['peak'] = close.rolling(p.peak_window).max()
        out['depth'] = 1.0 - close / out['peak'].replace(0, np.nan)
        out['boundary'] = out['depth'].shift(1).rolling(p.reference_window).quantile(p.tail_quantile)
        out['in_tail'] = (out['boundary'] > 0) & (out['depth'] >= out['boundary'])
        out['tail_dwell'] = out['in_tail'].astype(int).shift(1).rolling(p.dwell_bars).sum()
        prev_close = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - prev_close).abs(), (data['low'] - prev_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window).mean()
        out['entry_ready'] = ((out['tail_dwell'] == p.dwell_bars) & (out['depth'].shift(1) >= out['boundary']) & (out['depth'] < out['boundary']) & (out['peak'] == out['peak'].shift(1)) & (out['atr'] > 0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        p = params
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        pending = active = armed = False
        entry = stop = high_water = 0.0
        age = 0
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry = high_water = opens[i]
                    stop = entry - p.trail_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                high_water = max(high_water, closes[i])
                if closes[i] >= entry * (1.0 + p.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, high_water - p.trail_atr * atr[i])
                # Close-confirmed exits execute on the following bar.
                if closes[i] <= stop or age >= p.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
                pending = True
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    p = Params()
    index = pd.date_range('2020-01-01', periods=8)
    close = np.array([100, 100, 103, 104, 102.5, 101.5, 100, 100], dtype=float)
    data = pd.DataFrame({'open': 100.0, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'atr': [1, 1, 1, 1, 5, 1, 1, 1], 'entry_ready': [True, False, False, False, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, p).data
    # Expanding ATR must not lower the stop established at 102.
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    p.max_hold = 2
    assert strategy.generate_signals(data, ind, ctx, p).data['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    p.max_hold = 10
    data.loc[index[1], 'open'] = 105.0
    assert strategy.generate_signals(data, ind, ctx, p).data['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    actual = strategy.indicators(data, p)
    assert actual.index.equals(data.index)
    assert not strategy.generate_signals(data, actual, ctx, p).data['signal'].any()
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
