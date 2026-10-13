from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RatchetParams:
    compression_window: int = 5
    baseline_window: int = 20
    agreement_window: int = 3
    compression_ratio: float = 0.8
    gap_budget: float = 0.4
    body_pressure: float = 0.15
    profit_target: float = 0.025
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[RatchetParams]):
    strategy_id = 'gen_local-w2_1791294687'

    @classmethod
    def params_type(cls):
        return RatchetParams

    def warmup_bars(self, params: RatchetParams) -> int:
        return max(params.compression_window, params.baseline_window, params.agreement_window) + 1

    def indicators(self, data: pd.DataFrame, params: RatchetParams) -> pd.DataFrame:
        if not (1 <= params.compression_window <= params.baseline_window and params.agreement_window >= 1 and params.max_hold >= 1):
            raise ValueError('Invalid window or holding period')
        if not (0 < params.compression_ratio <= 1 and 0 < params.gap_budget < 1.2 and 0 < params.body_pressure < 1 and params.profit_target > 0):
            raise ValueError('Invalid signal or profit thresholds')
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['body'] = data['close'] - data['open']
        out['gap'] = data['open'] - data['close'].shift(1)
        out['recent_range'] = out['range'].rolling(params.compression_window).mean().shift(1)
        out['baseline_range'] = out['range'].rolling(params.baseline_window).mean().shift(1)
        out['compression'] = out['recent_range'] / out['baseline_range']
        out['gap_min'] = out['gap'].rolling(params.agreement_window).min()
        out['gap_max'] = out['gap'].rolling(params.agreement_window).max()
        out['gap_sum'] = out['gap'].rolling(params.agreement_window).sum()
        out['gap_load'] = out['gap_sum'] / out['recent_range']
        out['valid_volume'] = data['volume'].where(data['volume'] > 0)
        out['weighted_body'] = out['body'] * out['valid_volume']
        out['weighted_range'] = out['range'] * out['valid_volume']
        out['body_sum'] = out['weighted_body'].rolling(params.agreement_window).sum()
        out['range_sum'] = out['weighted_range'].rolling(params.agreement_window).sum()
        out['pressure'] = out['body_sum'] / out['range_sum'].where(out['range_sum'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RatchetParams) -> SignalFrame:
        compressed = indicators['compression'].le(params.compression_ratio)
        modest = indicators['gap_load'].abs().between(params.gap_budget, 1.2)
        long_entry = compressed & modest & indicators['gap_min'].gt(0) & indicators['pressure'].ge(params.body_pressure)
        short_entry = compressed & modest & indicators['gap_max'].lt(0) & indicators['pressure'].le(-params.body_pressure)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position:
                # The previous close's decision first fills at this bar's open.
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError('Invalid entry open')
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    entry_bar = -1
            elif i >= warmup - 1:
                position = int(entries[i])
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': [100.0, 100.0, 100.0, 103.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    params = RatchetParams(compression_window=1, baseline_window=1, agreement_window=1, max_hold=2)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=2)
    derived = pd.DataFrame({'compression': 0.5, 'gap_load': 0.5, 'gap_min': 1.0, 'gap_max': 1.0, 'pressure': 0.5}, index=index)
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].tolist() == [0, 0, 1, 1, 0, 1, 1]
    data['close'] = 100.0
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].tolist() == [0, 0, 1, 1, 0, 1, 1]
    derived['gap_min'] = -1.0
    derived['gap_max'] = -1.0
    derived['gap_load'] = -0.5
    derived['pressure'] = -0.5
    data.loc[index[2], 'close'] = 97.0
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].tolist() == [0, 0, -1, 0, -1, -1, 0]
    derived['pressure'] = 0.0
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].eq(0).all()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.generate_signals(data, actual, ctx, params).data['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
