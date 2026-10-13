from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    ma_bars: int = 20
    percentile_bars: int = 126
    entry_percentile: float = 0.85
    profit_target: float = 0.025
    max_hold_bars: int = 10

    def __post_init__(self):
        if self.ma_bars < 2 or self.percentile_bars < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('Entry percentile must be between 0.5 and 1')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('Profit target must be positive and finite')
        if self.max_hold_bars < 1:
            raise ValueError('Holding period must be positive')


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = 'gen_local-w2_1791233464'

    @classmethod
    def params_type(cls):
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return int(params.ma_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out['price_std'] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['price_std'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        history = out['z'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out['entry_threshold'] = history.quantile(params.entry_percentile)
        out['reset_threshold'] = history.quantile(0.5)
        out['entry'] = (
            (out['z'] > 0.0)
            & (out['z'] > out['entry_threshold'])
            & (out['z'].shift(1) <= out['entry_threshold'].shift(1))
        ).fillna(False)
        out['reset'] = (out['z'] <= out['reset_threshold']).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        resets = indicators['reset'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        armed = True
        holding = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if resets[i]:
                armed = True
            if holding:
                if i == entry_bar:
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or timed_out:
                    holding = False
                else:
                    signals[i] = 1
            elif armed and entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                holding = True
                armed = False
                entry_bar = i + 1
                entry_price = np.nan
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0,
        'high': 104.0,
        'low': 99.0,
        'close': [100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'entry': [True, False, False, True, False, False, False],
        'reset': [False, False, True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = BreakoutParams(profit_target=0.02, max_hold_bars=2)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    calculated = strategy.indicators(data, params)
    assert not calculated['entry'].any()
    assert strategy.warmup_bars(params) >= params.ma_bars + params.percentile_bars + 1


if __name__ == '__main__':
    _self_check()
