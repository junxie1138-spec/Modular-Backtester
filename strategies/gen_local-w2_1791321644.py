from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalBreakoutParams:
    fractional_order: float = 0.45
    fractional_window: int = 64
    breakout_window: int = 30
    noise_window: int = 20
    minimum_snr: float = 0.35
    profit_target: float = 0.045
    holding_bars: int = 18

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if min(self.fractional_window, self.breakout_window, self.noise_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if self.holding_bars < 1 or self.profit_target <= 0.0 or self.minimum_snr < 0.0:
            raise ValueError('invalid holding period, profit target, or SNR threshold')


class GeneratedStrategy(BaseStrategy[FractionalBreakoutParams]):
    strategy_id = 'gen_local-w2_1791321644'

    @classmethod
    def params_type(cls):
        return FractionalBreakoutParams

    def warmup_bars(self, params: FractionalBreakoutParams) -> int:
        return int(params.fractional_window + max(params.breakout_window + 1, params.noise_window + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalBreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(np.isfinite(close) & close.gt(0.0)))
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = out['log_close'].copy()
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * out['log_close'].shift(k)
        out['fractional_close'] = fractional_close
        out['breakout_level'] = fractional_close.rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        out['increment'] = fractional_close.diff()
        out['noise'] = out['increment'].rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0)
        out['snr'] = (out['increment'] / out['noise'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['qualified_breakout'] = (out['fractional_close'].gt(out['breakout_level']) & out['snr'].ge(params.minimum_snr)).fillna(False)
        out['confirmed_entry'] = out['qualified_breakout'] & out['qualified_breakout'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalBreakoutParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['confirmed_entry'].fillna(False).to_numpy(dtype=bool)
        pending_entry = False
        holding = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    holding = True
                    entry_price = opens[i]
                    entry_bar = i
            if holding:
                reached_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                reached_time_stop = i - entry_bar + 1 >= params.holding_bars
                if reached_target or reached_time_stop:
                    holding = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = 1
                pending_entry = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    sample = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'confirmed_entry': [False, False, True, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = FractionalBreakoutParams(holding_bars=3, profit_target=0.05)
    result = strategy.generate_signals(sample, entries, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    sample.loc[index[4], 'close'] = 106.0
    result = strategy.generate_signals(sample, entries, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
