from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    floor_ratio: float = 0.65

    def __post_init__(self):
        if self.lookback < 2 or not 0.0 < self.floor_ratio < 1.0:
            raise ValueError('Require lookback >= 2 and 0 < floor_ratio < 1')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791196180'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.lookback + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        for side in ('low', 'high'):
            prices = data[side].where(data[side] > 0)
            changes = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            out[side + '_change'] = changes
            window = changes.rolling(params.lookback, min_periods=params.lookback)
            out[side + '_drift'] = window.mean()
            out[side + '_vol'] = window.std(ddof=0)
        out['floor_ratio'] = out['low_vol'].div(out['high_vol'].where(out['high_vol'] > 0))
        out['previous_ratio'] = out['floor_ratio'].shift(1)
        out['entry'] = (
            out['floor_ratio'].lt(params.floor_ratio)
            & out['previous_ratio'].ge(params.floor_ratio)
            & out['low_drift'].gt(0)
            & out['high_drift'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, candidate in enumerate(entries):
            if remaining:
                signal[i] = 1
                remaining -= 1
            elif i and signal[i - 1] == 1:
                # Force the scheduled exit; ignore candidates on this bar.
                continue
            elif candidate:
                signal[i] = 1
                remaining = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    entries = pd.DataFrame({'entry': True}, index=index)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert result['size'].gt(0).all()
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
