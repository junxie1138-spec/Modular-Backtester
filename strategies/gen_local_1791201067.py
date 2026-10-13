from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    fast_window: int = 5
    slow_window: int = 30
    compression_limit: float = 0.75
    pressure_threshold: float = 0.20
    profit_target: float = 0.025
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = 'gen_local_1791201067'

    @classmethod
    def params_type(cls):
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        if not (2 <= params.fast_window <= params.slow_window):
            raise ValueError('Require 2 <= fast_window <= slow_window')
        if params.hold_bars < 1 or params.profit_target <= 0:
            raise ValueError('Holding period and profit target must be positive')
        if params.compression_limit <= 0 or params.pressure_threshold <= 0:
            raise ValueError('Compression and pressure thresholds must be positive')
        return int(params.slow_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        prior_close = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = (data['open'] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        out['intraday_range'] = ((data['high'] - data['low']) / prior_close).where(data['high'] >= data['low'])
        out['fast_range'] = out['intraday_range'].rolling(params.fast_window).mean()
        out['slow_range'] = out['intraday_range'].rolling(params.slow_window).mean()
        out['gap_sum'] = out['gap'].rolling(params.fast_window).sum()
        out['range_budget'] = out['intraday_range'].rolling(params.fast_window).sum()
        out['compression'] = out['fast_range'] / out['slow_range'].where(out['slow_range'] > 0)
        out['pressure'] = out['gap_sum'] / out['range_budget'].where(out['range_budget'] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        compressed = out['compression'].le(params.compression_limit)
        out['long_candidate'] = compressed & out['pressure'].ge(params.pressure_threshold) & out['gap'].gt(0) & data['close'].ge(data['open'])
        out['short_candidate'] = compressed & out['pressure'].le(-params.pressure_threshold) & out['gap'].lt(0) & data['close'].le(data['open'])
        out['long_confirmed'] = out['long_candidate'] & out['long_candidate'].shift(1, fill_value=False)
        out['short_confirmed'] = out['short_candidate'] & out['short_candidate'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        long_entries = indicators['long_confirmed'].fillna(False).to_numpy(dtype=bool)
        short_entries = indicators['short_confirmed'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # A previous close decision first becomes a position at today's open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                position = int(decisions[i - 1])
                entry_price = opens[i]
                entry_bar = i
            if position != 0:
                gain = position * (closes[i] / entry_price - 1)
                held_bars = i - entry_bar + 1
                # Close-observed targets exit at the following open, like all decisions.
                if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.hold_bars:
                    position = 0
                    continue
                decisions[i] = position
            elif np.isfinite(closes[i]) and closes[i] > 0:
                if long_entries[i]:
                    decisions[i] = 1
                elif short_entries[i]:
                    decisions[i] = -1
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapCapacityParams(hold_bars=2, profit_target=0.02)
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': [100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'long_confirmed': [False, True, False, False, False, False, False], 'short_confirmed': [False, False, False, True, False, False, False]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    computed = strategy.indicators(data, params)
    assert not computed['long_confirmed'].any()
    assert not computed['short_confirmed'].any()
    assert strategy.warmup_bars(params) >= params.slow_window + 1


if __name__ == '__main__':
    _self_check()
