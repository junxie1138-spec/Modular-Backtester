from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 5
    slow_window: int = 63
    weakness_percentile: float = 0.35
    atr_window: int = 14
    stop_atr: float = 1.5
    holding_bars: int = 4

    def __post_init__(self):
        if not 2 <= self.fast_window < self.slow_window:
            raise ValueError('Require 2 <= fast_window < slow_window')
        if self.atr_window < 1 or not 3 <= self.holding_bars <= 5:
            raise ValueError('Invalid ATR window or holding horizon')
        if not 0.0 < self.weakness_percentile < 0.5:
            raise ValueError('weakness_percentile must lie between zero and 0.5')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791340253'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slow_window, params.fast_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['fast_rank'] = close.rolling(params.fast_window, min_periods=params.fast_window).rank(pct=True)
        out['slow_rank'] = close.rolling(params.slow_window, min_periods=params.slow_window).rank(pct=True)
        out['previous_fast_rank'] = out['fast_rank'].shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['entry'] = (
            (out['previous_fast_rank'] <= 0.5)
            & (out['fast_rank'] > 0.5)
            & (out['slow_rank'] <= params.weakness_percentile)
            & np.isfinite(out['atr'])
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        high_water = stop = -np.inf
        age = 0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # First held close occurs after the entry decision bar.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.stop_atr * atr)
                    if close <= stop or age >= params.holding_bars:
                        active = False
            elif valid and entries[i]:
                active = True
                high_water = stop = -np.inf
                age = 0
            decisions[i] = int(active)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0, 101.0, 103.0, 100.0, 101.0]}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 10.0, 1.0], 'entry': [True, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    indicators['atr'] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params(holding_bars=3)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    indicators['atr'] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result['signal'].eq(0).all()
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
