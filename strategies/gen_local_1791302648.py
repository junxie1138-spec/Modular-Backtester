from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 63
    tail_fraction: float = 0.2
    atr_window: int = 14
    atr_multiple: float = 2.5
    holding_bars: int = 8

    def __post_init__(self):
        if self.rank_window < 2 or self.atr_window < 1:
            raise ValueError('Lookback windows must be positive; rank_window must exceed one.')
        if not 0.0 < self.tail_fraction < 0.5:
            raise ValueError('tail_fraction must lie between zero and one half.')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive.')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791302648'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.rank_window + 5, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return5'] = close.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['percentile'] = out['return5'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['mid_month'] = (data.index.day >= 10) & (data.index.day <= 16)
        valid = np.isfinite(out[['atr', 'ma200', 'percentile', 'return5']]).all(axis=1) & (out['atr'] > 0)
        buy = valid & out['mid_month'] & (close > out['ma200']) & (out['return5'] < 0) & (out['percentile'] <= params.tail_fraction)
        sell = valid & out['mid_month'] & (close < out['ma200']) & (out['return5'] > 0) & (out['percentile'] >= 1.0 - params.tail_fraction)
        out['entry'] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = lowest = lower_stop = upper_stop = np.nan
        rows = zip(data['close'].to_numpy(dtype=float), indicators['atr'].to_numpy(dtype=float), indicators['ma200'].to_numpy(dtype=float), indicators['entry'].to_numpy())
        for i, (close, atr, ma, entry) in enumerate(rows):
            if not np.isfinite([close, atr, ma]).all() or close <= 0 or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                # Entry fills on this bar after the previous close's decision.
                if age == 0:
                    highest = lowest = close
                    lower_stop = close - params.atr_multiple * atr
                    upper_stop = close + params.atr_multiple * atr
                else:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    lower_stop = max(lower_stop, highest - params.atr_multiple * atr)
                    upper_stop = min(upper_stop, lowest + params.atr_multiple * atr)
                age += 1
                # The required high-water stop also harvests short-side declines.
                high_water_exit = close <= lower_stop
                short_rebound_exit = position == -1 and close >= upper_stop
                regime_exit = (position == 1 and close <= ma) or (position == -1 and close >= ma)
                if high_water_exit or short_rebound_exit or regime_exit or age >= params.holding_bars:
                    position = 0
                    age = 0
                    continue
            elif entry != 0:
                position = int(entry)
                age = 0
            targets[i] = position
        df = pd.DataFrame({'signal': targets, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    params = Params(atr_multiple=2.0)
    data = pd.DataFrame({'close': [100.0, 100.0, 105.0, 102.0, 101.0]}, index=index)
    ind = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 10.0, 10.0], 'ma200': 90.0, 'entry': [1, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    data['close'] = [100.0, 100.0, 99.0, 102.0, 102.0]
    ind['atr'] = 1.0
    ind['ma200'] = 110.0
    ind['entry'] = [-1, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, 0]
    assert (result['size'] > 0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
