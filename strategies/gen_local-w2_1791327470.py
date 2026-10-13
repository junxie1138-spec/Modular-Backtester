from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalCompressionParams:
    fast_window: int = 3
    slow_window: int = 20
    compression_ratio: float = 0.85
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalCompressionParams]):
    strategy_id = 'gen_local-w2_1791327470'

    @classmethod
    def params_type(cls):
        return SeasonalCompressionParams

    def warmup_bars(self, params: SeasonalCompressionParams) -> int:
        return int(max(params.fast_window, params.slow_window) + 2)

    def indicators(self, data: pd.DataFrame, params: SeasonalCompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['fast_atr'] = out['true_range'].rolling(params.fast_window).mean()
        out['atr'] = out['true_range'].rolling(params.slow_window).mean()
        out['compression'] = out['fast_atr'].div(out['atr'].where(out['atr'] > 0))
        out['weekday'] = data.index.dayofweek
        out['quiet'] = out['compression'].le(params.compression_ratio).fillna(False)
        out['confirmed'] = (
            out['quiet']
            & out['quiet'].shift(1, fill_value=False)
            & out['weekday'].eq(1)
            & out['weekday'].shift(1).eq(0)
            & data['close'].ge(previous_close)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalCompressionParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            # The previous close's entry decision fills at this bar's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                entry = opens[i]
                prior_atr = atrs[i - 1]
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(prior_atr) or prior_atr <= 0:
                    continue
                active = True
                armed = False
                age = 0
                peak = entry
                stop = entry - params.initial_stop_atr * prior_atr
            if active:
                age += 1
                close = closes[i]
                atr = atrs[i]
                # ponytail: close-only stops; intrabar fills need execution-engine support.
                if not np.isfinite(close) or close <= stop or age >= params.holding_bars:
                    active = False
                    continue
                peak = max(peak, close)
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, peak - params.trail_atr * atr)
                    if close <= stop:
                        active = False
                        continue
                decisions[i] = 1
            elif confirmed[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = SeasonalCompressionParams(holding_bars=5, trail_atr=1.0)
    index = pd.bdate_range('2024-01-01', periods=8)
    close = np.array([100., 100., 100., 102., 100.5, 100., 100., 100.])
    bars = pd.DataFrame({'open': 100., 'high': np.maximum(close, 100.) + 1.,
                         'low': np.minimum(close, 100.) - 1., 'close': close,
                         'volume': 1000.}, index=index)
    derived = pd.DataFrame({'atr': 1., 'confirmed': [False, True, False, False,
                                                    False, False, False, False]}, index=index)
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    params.holding_bars = 3
    flat = bars.copy()
    flat['close'] = 100.
    result = strategy.generate_signals(flat, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    params.fast_window = 2
    params.slow_window = 2
    params.compression_ratio = 1.01
    flat['high'] = 101.
    flat['low'] = 99.
    actual = strategy.indicators(flat, params)
    assert actual['confirmed'].tolist() == [False, False, False, False, False, False, True, False]
