from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarDrawdownParams:
    lookback: int = 63
    deep_drawdown: float = 0.025
    shallow_drawdown: float = 0.0075
    profit_target: float = 0.018
    max_hold: int = 8

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError('lookback must be an integer >= 2')
        if not isinstance(self.max_hold, int) or self.max_hold < 1:
            raise ValueError('max_hold must be a positive integer')
        values = [self.deep_drawdown, self.shallow_drawdown, self.profit_target]
        if not np.isfinite(values).all():
            raise ValueError('thresholds must be finite')
        if not 0 < self.shallow_drawdown < self.deep_drawdown < 1:
            raise ValueError('require 0 < shallow_drawdown < deep_drawdown < 1')
        if not 0 < self.profit_target < 1:
            raise ValueError('profit_target must be between zero and one')


class GeneratedStrategy(BaseStrategy[CalendarDrawdownParams]):
    strategy_id = 'gen_local-w2_1790968959'

    @classmethod
    def params_type(cls):
        return CalendarDrawdownParams

    @staticmethod
    def warmup_bars(params):
        return int(params.lookback)

    @staticmethod
    def indicators(data, params):
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & (close > 0))
        peak = close.rolling(params.lookback, min_periods=params.lookback).max()
        depth = (1.0 - close / peak).clip(0.0, 1.0)
        day = pd.Series(data.index.day, index=data.index)
        boundary = (day <= 5) | (day >= 25)
        midmonth = (day >= 10) & (day <= 20)
        ready = depth.notna()
        buy = ready & boundary & (depth >= params.deep_drawdown)
        sell = ready & midmonth & (depth <= params.shallow_drawdown)
        candidate = pd.Series(np.select([buy, sell], [1, -1], default=0), index=data.index, dtype=int)
        long_strength = (depth / (2.0 * params.deep_drawdown)).clip(0.0, 1.0).fillna(0.0)
        short_strength = (1.0 - depth / params.shallow_drawdown).clip(0.0, 1.0).fillna(0.0)
        strength = pd.Series(np.select([buy, sell], [long_strength, short_strength], default=0.0), index=data.index)
        return pd.DataFrame({
            'peak': peak.fillna(0.0),
            'depth': depth.fillna(0.0),
            'day': day,
            'boundary': boundary,
            'midmonth': midmonth,
            'ready': ready,
            'long_strength': long_strength,
            'short_strength': short_strength,
            'strength': strength,
            'candidate': candidate,
            'entry_size': 0.25 + 0.70 * strength,
        }, index=data.index)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        proposed_sizes = indicators['entry_size'].fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        position_size = 0.25
        for i in range(n):
            if position:
                # The preceding close's instruction fills at this bar's open.
                if entry_bar == -1:
                    entry_price = opens[i]
                    entry_bar = i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
                    # Exit orders are emitted below; no same-close reentry.
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                position_size = float(proposed_sizes[i])
                entry_bar = -1
            signals[i] = position
            sizes[i] = position_size if position else 0.25
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': [100.0, 102.0, 100.0, 100.0, 98.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'candidate': [1, 0, 0, -1, 0, 0, 0], 'entry_size': 0.60}, index=index)
    params = CalendarDrawdownParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
    assert result.loc[index[1], 'size'] == 0.60
    data['close'] = 100.0
    indicators['candidate'] = [1, 0, 0, 0, 0, 0, 0]
    params = CalendarDrawdownParams(max_hold=2)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
