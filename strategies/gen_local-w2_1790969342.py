from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    depth_capacity: float = 3.0
    profit_target: float = 0.006

    def __post_init__(self):
        if not np.isfinite(self.depth_capacity) or self.depth_capacity <= 0:
            raise ValueError('depth_capacity must be finite and positive')
        if not np.isfinite(self.profit_target) or not 0 < self.profit_target < 1:
            raise ValueError('profit_target must be between zero and one')


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = 'gen_local-w2_1790969342'

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return 23

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['absolute_return'] = close.pct_change(fill_method=None).abs()
        out['daily_capacity'] = out['absolute_return'].rolling(20, min_periods=20).mean().shift(1)
        out['prior_peak'] = close.rolling(10, min_periods=10).max().shift(1)
        out['prior_trough'] = close.rolling(10, min_periods=10).min().shift(1)
        out['normal_volume'] = volume.rolling(20, min_periods=20).median().shift(1)
        out['drawdown'] = (1.0 - close / out['prior_peak']).clip(lower=0)
        out['drawup'] = (close / out['prior_trough'] - 1.0).clip(lower=0)
        capacity = out['daily_capacity'].where(out['daily_capacity'] > 0)
        out['down_load'] = out['drawdown'] / capacity
        out['up_load'] = out['drawup'] / capacity
        out['previous_down_load'] = out['down_load'].shift(1)
        out['previous_up_load'] = out['up_load'].shift(1)
        out['close_change'] = close.diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & close.notna()
        out['thin_volume'] = ((volume > 0) & (volume < out['normal_volume'])).fillna(False)
        long_entry = (
            ready & out['thin_volume']
            & (out['down_load'] >= params.depth_capacity)
            & (out['previous_down_load'] < params.depth_capacity)
            & (out['close_change'] < 0)
        )
        short_entry = (
            ready & out['thin_volume']
            & (out['up_load'] >= params.depth_capacity)
            & (out['previous_up_load'] < params.depth_capacity)
            & (out['close_change'] > 0)
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # Observe the entry open only once its bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_price and valid_close else -np.inf
                held_bars = i - entry_bar + 1
                # Close-observed targets execute at the following open.
                if gain >= params.profit_target or held_bars >= 2 or not valid_price or not valid_close:
                    side = 0
                    entry_price = np.nan
            elif entries[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry_bar = i + 1
            decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=9, freq='D')
    data = pd.DataFrame({
        'open': 100.0, 'high': 102.0, 'low': 98.0,
        'close': [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0, -1, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, CapacityParams()).data
    # Long target after one session, short time-stop after two, short target after one.
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, -1, 0, -1, 0]
    assert result['size'].gt(0).all()
    flat = data.assign(close=100.0, high=100.0, low=100.0)
    calculated = GeneratedStrategy.indicators(None, flat, CapacityParams())
    assert calculated['entry'].eq(0).all()


if __name__ == '__main__':
    _self_check()
