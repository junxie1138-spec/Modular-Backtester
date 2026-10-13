from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    recent_window: int = 21
    peak_window: int = 63
    percentile_window: int = 252
    depth_quantile: float = 0.80
    separation_tail: float = 0.25
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791246065'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.recent_window, params.peak_window) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (1 < params.recent_window < params.peak_window and params.percentile_window >= 2):
            raise ValueError('Require 1 < recent_window < peak_window and percentile_window >= 2')
        if not (0 < params.depth_quantile < 1 and 0 < params.separation_tail < 0.5):
            raise ValueError('Invalid percentile parameters')
        if params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError('Exit parameters must be positive')
        close = data['close'].where(np.isfinite(data['close']) & (data['close'] > 0))
        out = pd.DataFrame(index=data.index)
        out['recent_peak'] = close.rolling(params.recent_window, min_periods=params.recent_window).max()
        out['older_peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['recent_depth'] = 1.0 - close / out['recent_peak']
        out['depth'] = 1.0 - close / out['older_peak']
        out['separation'] = out['depth'] - out['recent_depth']
        out['depth_change'] = out['depth'].diff()
        depth_history = out['depth'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        separation_history = out['separation'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        out['depth_threshold'] = depth_history.quantile(params.depth_quantile)
        out['separation_low'] = separation_history.quantile(params.separation_tail)
        out['separation_high'] = separation_history.quantile(1.0 - params.separation_tail)
        ready = out.notna().all(axis=1)
        deep = ready & (out['depth'] > out['depth_threshold'])
        recovering = deep & (out['separation'] > out['separation_high']) & (out['depth_change'] < 0)
        deteriorating = deep & (out['separation'] <= out['separation_low']) & (out['depth_change'] > 0)
        out['entry'] = np.select([recovering, deteriorating], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                # Observe the execution open only when its bar has arrived.
                if entry_bar == i:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                if position != 0:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if (np.isfinite(gain) and gain >= params.profit_target) or i - entry_bar + 1 >= params.max_hold:
                        position = 0
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 94.0, 'close': [100.0, 105.0, 100.0, 100.0, 95.0, 100.0], 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, entries, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0]
    data['close'] = 100.0
    entries['entry'] = [1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, entries, None, Params(max_hold=2)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
