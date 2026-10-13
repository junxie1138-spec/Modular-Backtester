from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    lookback: int = 20
    entropy_ceiling: float = 0.88
    anticorrelation_floor: float = 0.12
    profit_target: float = 0.008
    max_hold_bars: int = 2

    def __post_init__(self):
        if self.lookback < 6:
            raise ValueError('lookback must be at least 6')
        if not 0.0 < self.entropy_ceiling <= 1.0:
            raise ValueError('entropy_ceiling must be in (0, 1]')
        if not 0.0 < self.anticorrelation_floor <= 1.0:
            raise ValueError('anticorrelation_floor must be in (0, 1]')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be finite and positive')
        if self.max_hold_bars not in (1, 2):
            raise ValueError('max_hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local-w2_1791260564'

    @classmethod
    def params_type(cls):
        return OrdinalParams

    @staticmethod
    def warmup_bars(params: OrdinalParams) -> int:
        return int(params.lookback + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
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
            column = f'pattern_probability_{code}'
            out[column] = out['pattern'].eq(code).astype(float).where(valid).rolling(params.lookback, min_periods=params.lookback).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        out['ordinal_entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-15))).sum(axis=1, min_count=6) / np.log(6.0)
        out['lag1_correlation'] = out['return'].rolling(params.lookback, min_periods=params.lookback).corr(out['return_lag1']).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out['ordinal_agrees'] = out['ordinal_entropy'].le(params.entropy_ceiling) & out['ordinal_entropy'].notna()
        out['autocorrelation_agrees'] = out['lag1_correlation'].le(-params.anticorrelation_floor) & out['return'].lt(0.0)
        out['entry'] = (out['ordinal_agrees'] & out['autocorrelation_agrees']).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending_entry = False
        active = False
        entry_price = 0.0
        held_bars = 0
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                # The preceding close's decision fills at this observed open.
                entry_price = opens[i]
                held_bars = 0
                active = True
            if active:
                held_bars += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                # Close-observed exits execute at the following bar's open.
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                signals[i] = 1
                pending_entry = True
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'entry': [True, False, False, False]}, index=index)
    params = OrdinalParams()
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    data.loc[index[1], 'close'] = 101.0
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    data['close'] = 100.0
    result = GeneratedStrategy.generate_signals(data, features, None, OrdinalParams(max_hold_bars=1)).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    derived = GeneratedStrategy.indicators(data, params)
    assert not derived['entry'].any()
    assert (result['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
