from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRetestParams:
    peak_lookback: int = 63
    atr_period: int = 14
    minimum_depth: float = 0.04
    recovery_fraction: float = 0.5
    retest_fraction: float = 0.75
    atr_multiple: float = 3.0
    holding_bars: int = 20
    setup_bars: int = 30

    def __post_init__(self):
        if self.peak_lookback < 2 or self.atr_period < 1:
            raise ValueError('Lookbacks must be positive, with peak_lookback >= 2')
        if not 0 < self.minimum_depth < 1:
            raise ValueError('minimum_depth must lie between zero and one')
        if not 0 < self.recovery_fraction < self.retest_fraction < 1:
            raise ValueError('Require 0 < recovery_fraction < retest_fraction < 1')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive')
        if self.holding_bars < 1 or self.setup_bars < 1:
            raise ValueError('Holding and setup durations must be positive')


class GeneratedStrategy(BaseStrategy[DrawdownRetestParams]):
    strategy_id = 'gen_local-w2_1791195299'

    @classmethod
    def params_type(cls):
        return DrawdownRetestParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.peak_lookback, params.atr_period + 1))

    @staticmethod
    def indicators(data, params):
        close = data['close'].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out = pd.DataFrame(index=data.index)
        out['peak'] = close.rolling(params.peak_lookback, min_periods=params.peak_lookback).max()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['atr'] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        prices = data['close'].to_numpy(dtype=float)
        values = indicators[['peak', 'drawdown', 'ma200', 'atr']].to_numpy(dtype=float)
        active = False
        stage = 0
        anchor = np.nan
        deepest = 0.0
        setup_start = -1
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf

        for i in range(count):
            close = prices[i]
            peak, depth, average, atr = values[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(values[i]).all() and atr > 0
            if not valid:
                active = False
                stage = 0
                anchor = np.nan
                continue

            if active:
                # Entry decided on the previous close fills at this bar's open.
                # Only closes from the filled position contribute to its high.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.atr_multiple * atr)
                if close <= stop or close <= average or i - entry_bar >= params.holding_bars:
                    active = False
                    stage = 0
                    anchor = np.nan
                else:
                    signals[i] = 1
                continue

            if close <= average:
                stage = 0
                anchor = np.nan
                continue

            if peak != anchor:
                anchor = peak
                stage = 0
            if stage and i - setup_start > params.setup_bars:
                stage = 0

            if stage == 0:
                if depth >= params.minimum_depth:
                    deepest = depth
                    setup_start = i
                    stage = 1
            elif stage == 1:
                deepest = max(deepest, depth)
                if depth <= params.recovery_fraction * deepest:
                    stage = 2
            elif depth >= deepest:
                deepest = depth
                setup_start = i
                stage = 1
            elif depth >= params.retest_fraction * deepest:
                active = True
                entry_bar = i
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
                stage = 0

        df = pd.DataFrame({'signal': signals, 'size': np.ones(count, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = DrawdownRetestParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [95.0, 94.0, 97.5, 95.5, 97.0, 100.0, 96.0, 95.0]}, index=index)
    ind = pd.DataFrame({'peak': 100.0, 'drawdown': [0.05, 0.06, 0.025, 0.045, 0.03, 0.0, 0.04, 0.05], 'ma200': 90.0, 'atr': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    prefix = GeneratedStrategy.generate_signals(data.iloc[:6], ind.iloc[:6], None, params).data
    assert prefix.equals(result.iloc[:6])
    timed = GeneratedStrategy.generate_signals(data, ind, None, DrawdownRetestParams(holding_bars=2)).data
    assert timed['signal'].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    blocked = ind.copy()
    blocked['ma200'] = 110.0
    assert GeneratedStrategy.generate_signals(data, blocked, None, params).data['signal'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
