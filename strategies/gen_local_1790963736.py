from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 8
    rank_window: int = 60
    record_window: int = 3
    efficiency_percentile: float = 0.80
    record_threshold: float = 0.35
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8

    def __post_init__(self):
        for value in (self.trend_window, self.rank_window, self.record_window, self.atr_window, self.holding_bars):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Windows and holding_bars must be positive integers')
        if not 0.0 < self.efficiency_percentile <= 1.0:
            raise ValueError('efficiency_percentile must be in (0, 1]')
        if not 0.0 < self.record_threshold <= 1.0:
            raise ValueError('record_threshold must be in (0, 1]')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790963736'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_window + params.rank_window, params.record_window + params.trend_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['change'] = close.diff()
        out['net_move'] = out['change'].rolling(params.trend_window).sum()
        out['path_length'] = out['change'].abs().rolling(params.trend_window).sum()
        out['efficiency'] = out['net_move'].div(out['path_length'].replace(0.0, np.nan))
        out['efficiency_rank'] = out['efficiency'].abs().rolling(params.rank_window).rank(pct=True)
        out['previous_high'] = data['high'].shift(1).rolling(params.record_window).max()
        out['previous_low'] = data['low'].shift(1).rolling(params.record_window).min()
        valid_records = out['previous_high'].notna() & out['previous_low'].notna() & data['high'].notna() & data['low'].notna()
        out['record_vote'] = ((data['high'] > out['previous_high']).astype(float) - (data['low'] < out['previous_low']).astype(float)).where(valid_records)
        out['record_pressure'] = out['record_vote'].rolling(params.trend_window).mean()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - out['previous_close']).abs(), (data['low'] - out['previous_close']).abs()], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        strong = indicators['efficiency_rank'].ge(params.efficiency_percentile)
        long_entry = strong & indicators['efficiency'].gt(0.0) & indicators['record_pressure'].ge(params.record_threshold)
        short_entry = strong & indicators['efficiency'].lt(0.0) & indicators['record_pressure'].le(-params.record_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop_distance = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # The preceding close's order fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                invalid_price = not np.isfinite(entry) or entry <= 0.0 or not np.isfinite(closes[i]) or closes[i] <= 0.0
                stopped = position * (closes[i] - entry) < -stop_distance
                if invalid_price or stopped or age >= params.holding_bars:
                    position = 0
                    entry = np.nan
                    age = 0
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(candidates[i])
                entry = np.nan
                # Freeze decision-bar ATR for the entire trade.
                stop_distance = params.stop_atr * atr[i]
                age = 0
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=5)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    for side in (1, -1):
        data = pd.DataFrame({'open': [100.0, 100.0 + side * 10.0, 100.0, 100.0, 100.0], 'close': [100.0, 100.0 + side * 9.0, 100.0 + side * 7.0, 100.0, 100.0]}, index=index)
        indicators = pd.DataFrame({'efficiency_rank': [1.0, 0.0, 0.0, 0.0, 0.0], 'efficiency': float(side), 'record_pressure': float(side), 'atr': [1.0, 20.0, 20.0, 20.0, 20.0]}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result['signal'].tolist() == [0, side, side, 0, 0]
        assert result['size'].gt(0.0).all()
        data['open'] = 100.0
        data['close'] = 100.0
        timed = strategy.generate_signals(data, indicators, ctx, Params(holding_bars=2)).data
        assert timed['signal'].tolist() == [0, side, side, 0, 0]
    assert strategy.warmup_bars(params) >= 68


if __name__ == '__main__':
    _self_check()
