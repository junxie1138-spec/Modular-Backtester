from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapReclamationParams:
    lookback: int = 20
    roc_bars: int = 2
    gap_threshold: float = 1.1
    acceleration_threshold: float = 0.3
    profit_target: float = 0.015
    max_hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapReclamationParams]):
    strategy_id = 'gen_local_1790924314'

    @classmethod
    def params_type(cls):
        return GapReclamationParams

    def warmup_bars(self, params: GapReclamationParams) -> int:
        return int(max(params.lookback + 2, params.roc_bars + 2))

    def indicators(self, data: pd.DataFrame, params: GapReclamationParams) -> pd.DataFrame:
        if params.lookback < 2 or params.roc_bars < 1 or params.max_hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; lookback must be at least two.')
        if not (params.gap_threshold > 0 and params.acceleration_threshold > 0 and params.profit_target > 0):
            raise ValueError('Signal thresholds and profit target must be positive.')
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError('Position sizes must satisfy 0 < min_size <= max_size <= 1.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        previous_close = close.shift(1)
        out['gap'] = opening.div(previous_close).sub(1)
        out['gap_scale'] = out['gap'].rolling(params.lookback, min_periods=params.lookback).std(ddof=0).shift(1).replace(0, np.nan)
        out['return'] = close.div(previous_close).sub(1)
        out['return_scale'] = out['return'].rolling(params.lookback, min_periods=params.lookback).std(ddof=0).shift(1).replace(0, np.nan)
        out['roc'] = close.div(close.shift(params.roc_bars)).sub(1)
        out['acceleration'] = out['roc'].diff()
        out['strength'] = out['acceleration'].div(out['return_scale'])
        upward_failure = (out['gap'] > params.gap_threshold * out['gap_scale']) & (close <= previous_close)
        downward_failure = (out['gap'] < -params.gap_threshold * out['gap_scale']) & (close >= previous_close)
        out['failed_gap'] = np.select([upward_failure, downward_failure], [1, -1], default=0)
        out['prior_failed_gap'] = out['failed_gap'].shift(1).fillna(0)
        out['reclaimed'] = np.select([close > opening.shift(1), close < opening.shift(1)], [1, -1], default=0)
        out = out.replace([np.inf, -np.inf], np.nan)
        qualified = (out['prior_failed_gap'] != 0) & (out['reclaimed'] == out['prior_failed_gap']) & (out['strength'] * out['prior_failed_gap'] >= params.acceleration_threshold)
        out['entry'] = out['prior_failed_gap'].where(qualified, 0).fillna(0).astype(int)
        out['entry_size'] = (params.min_size + (params.max_size - params.min_size) * (out['strength'].abs() / (3 * params.acceleration_threshold)).clip(0, 1)).fillna(params.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapReclamationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        openings = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators['entry_size'].fillna(params.min_size).to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        held_size = params.min_size
        for i in range(count):
            if side:
                # Yesterday's decision fills at today's open; exits also fill next open.
                if i == entry_bar:
                    entry_price = openings[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = side * (closes[i] / entry_price - 1) if valid else 0.0
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    side = 0
                    held_size = params.min_size
            elif entries[i] and i + 1 < count:
                side = int(entries[i])
                entry_bar = i + 1
                held_size = float(entry_sizes[i])
            signals[i] = side
            sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = GapReclamationParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    manual = pd.DataFrame({'entry': [1, 0, 0, 0, 0, 0, 0, 0], 'entry_size': 0.7}, index=index)
    context = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    data.loc[index[2], 'close'] = 102.0
    result = strategy.generate_signals(data, manual, context, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result.loc[index[1], 'size'] == 0.7
    data['close'] = 100.0
    manual.loc[index[0], 'entry'] = -1
    result = strategy.generate_signals(data, manual, context, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, -1, 0, 0, 0]
    assert strategy.indicators(data, params)['entry'].eq(0).all()
