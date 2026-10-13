from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietBreakoutParams:
    breakout_lookback: int = 10
    percentile_window: int = 80
    range_quantile: float = 0.35
    atr_window: int = 14
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[QuietBreakoutParams]):
    strategy_id = 'gen_local-w2_1791319909'

    @classmethod
    def params_type(cls) -> type[QuietBreakoutParams]:
        return QuietBreakoutParams

    @staticmethod
    def warmup_bars(params: QuietBreakoutParams) -> int:
        return int(max(params.percentile_window + 2, params.breakout_lookback + 1, params.atr_window + 1, 4))

    @staticmethod
    def indicators(data: pd.DataFrame, params: QuietBreakoutParams) -> pd.DataFrame:
        if min(params.breakout_lookback, params.percentile_window, params.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.range_quantile < 1.0:
            raise ValueError('range_quantile must lie strictly between zero and one.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError('trail_atr must be finite and positive.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')

        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['daily_range'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['range_fraction'] = out['daily_range'] / out['previous_close'].where(out['previous_close'] > 0.0)
        out['range_fraction'] = out['range_fraction'].replace([np.inf, -np.inf], np.nan)
        out['range_threshold'] = out['range_fraction'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.range_quantile)
        out['prior_high'] = data['high'].shift(1).rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max()
        out['true_range'] = pd.concat([
            out['daily_range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['narrowing'] = (
            (out['range_fraction'] < out['range_fraction'].shift(1))
            & (out['range_fraction'].shift(1) < out['range_fraction'].shift(2))
        )
        ready = np.isfinite(out[['range_fraction', 'range_threshold', 'prior_high', 'atr']]).all(axis=1)
        out['entry'] = (
            ready
            & (out['atr'] > 0.0)
            & out['narrowing']
            & (out['range_fraction'] < out['range_threshold'])
            & (data['open'] <= out['prior_high'])
            & (data['close'] > out['prior_high'])
            & (data['close'] > data['open'])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuietBreakoutParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entered = -1
        high_water = stop = -np.inf

        for i, (close, atr, entry) in enumerate(zip(closes, atrs, entries)):
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0.0:
                active = False
            elif active:
                # The first in-trade close occurs after the entry decision bar.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or i - entered >= params.hold_bars:
                    active = False
            elif entry:
                active = True
                entered = i
                high_water = stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = QuietBreakoutParams()
    index = pd.date_range('2020-01-01', periods=6)
    cases = [
        ([100, 103, 102, 101, 100, 100], [1, 1, 10, 1, 1, 1], [0, 1, 1, 1, 0, 0]),
        ([100, 101, 102, 103, 104, 105], [1] * 6, [0, 1, 1, 1, 1, 0]),
    ]
    for closes, atrs, expected in cases:
        data = pd.DataFrame({'close': closes}, index=index)
        derived = pd.DataFrame({'atr': atrs, 'entry': [True, False, False, False, False, False]}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params)
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0.0).all()
