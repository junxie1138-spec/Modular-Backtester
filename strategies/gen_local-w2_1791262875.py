from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_bars: int = 63
    shelf_bars: int = 3
    atr_bars: int = 14
    min_depth: float = 0.025
    shelf_atr: float = 0.65
    breakeven_gain: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        if self.peak_bars < 2 or self.shelf_bars < 2 or self.atr_bars < 1:
            raise ValueError('Invalid indicator lookback')
        if not 0 < self.min_depth < 1:
            raise ValueError('min_depth must be between zero and one')
        if not all(np.isfinite(x) and x > 0 for x in (self.shelf_atr, self.breakeven_gain, self.trail_atr)):
            raise ValueError('Thresholds must be finite and positive')
        if not 5 <= self.max_hold <= 10:
            raise ValueError('max_hold must be five to ten sessions')


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = 'gen_local-w2_1791262875'

    @classmethod
    def params_type(cls):
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        return int(max(200, params.peak_bars + params.shelf_bars - 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['peak'] = data['high'].rolling(params.peak_bars, min_periods=params.peak_bars).max()
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['depth'] = 1.0 - data['close'] / out['peak'].replace(0, np.nan)
        out['shelf_min'] = out['depth'].rolling(params.shelf_bars, min_periods=params.shelf_bars).min()
        out['shelf_max'] = out['depth'].rolling(params.shelf_bars, min_periods=params.shelf_bars).max()
        out['depth_width'] = out['shelf_max'] - out['shelf_min']
        out['width_limit'] = params.shelf_atr * out['atr'] / out['peak'].replace(0, np.nan)
        out['shelf'] = (
            (out['shelf_min'] >= params.min_depth)
            & (out['depth_width'] <= out['width_limit'])
            & (out['atr'] > 0)
        ).fillna(False)
        out['entry'] = (
            out['shelf']
            & ~out['shelf'].shift(1, fill_value=False)
            & (data['close'] > out['ma200'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShelfParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        active = False
        armed = False
        entry_price = stop = highest_close = 0.0
        entry_bar = -1
        for i in range(len(data)):
            # The preceding close's decision enters at this bar's open.
            if not active and i > 0 and signals[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry_bar = i
                    entry_price = opens[i]
                    highest_close = entry_price
                    stop = entry_price - params.trail_atr * atr[i - 1]
            if active:
                if not all(np.isfinite(x) for x in (closes[i], lows[i], atr[i])) or atr[i] <= 0:
                    active = False
                    continue
                # Existing stops are observed during the bar; liquidation is next open.
                if lows[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    continue
                highest_close = max(highest_close, closes[i])
                if closes[i] >= entry_price * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, highest_close - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    sample = pd.DataFrame({
        'open': [100.0, 100.0, 102.0, 101.5, 99.0, 100.0],
        'high': [101.0, 103.0, 103.0, 102.0, 101.0, 101.0],
        'low': [99.0, 100.0, 101.0, 98.0, 98.0, 99.0],
        'close': [100.0, 102.0, 101.5, 99.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({'entry': [True, False, False, False, False, False], 'atr': 1.0}, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, ShelfParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result['size'] > 0).all()
