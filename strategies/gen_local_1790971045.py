from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 20
    entropy_ceiling: float = 0.85
    profit_target: float = 0.04
    holding_bars: int = 18

    def __post_init__(self):
        if self.complexity_window < 6:
            raise ValueError('complexity_window must be at least 6')
        if not 0.0 < self.entropy_ceiling < 1.0:
            raise ValueError('entropy_ceiling must be between zero and one')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be finite and positive')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790971045'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.complexity_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_lag1'] = out['return'].shift(1)
        out['return_lag2'] = out['return'].shift(2)
        a, b, c = out['return_lag2'], out['return_lag1'], out['return']
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out['pattern'] = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = 'probability_' + str(code)
            out[column] = out['pattern'].eq(code).astype(float).where(valid).rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        out['entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out['payroll_window'] = (data.index.day >= 10) & (data.index.day <= 14)
        out['entry_ready'] = out['entropy'].le(params.entropy_ceiling).fillna(False) & out['payroll_window']
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                # A prior close's entry decision fills at this session's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                target_hit = np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.holding_bars:
                    active = False
                    held = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                held = 0
            decisions[i] = int(active)
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = Params(holding_bars=3)
    index = pd.date_range('2020-01-10', periods=6, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_ready': [True, False, False, False, False, False]}, index=index)
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[2], 'close'] = 105.0
    targeted = strategy.generate_signals(data, indicators, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.warmup_bars(params) == 23
    assert not strategy.indicators(data, params)['entry_ready'].any()
