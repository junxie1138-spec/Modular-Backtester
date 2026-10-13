from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    flow_window: int = 10
    rank_window: int = 63
    tail: float = 0.20
    profit_target: float = 0.025
    max_hold: int = 8

    def __post_init__(self):
        if self.flow_window < 2 or self.rank_window < 2 or self.max_hold < 1:
            raise ValueError('Windows must be at least two and max_hold positive')
        if not 0.0 < self.tail < 0.5:
            raise ValueError('tail must be between zero and one-half')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be finite and positive')


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = 'gen_local-w2_1791236102'

    @classmethod
    def params_type(cls):
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(params.flow_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        volume = data['volume'].where(data['volume'] >= 0.0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_volume'] = out['return'] * volume
        out['volume_sum'] = volume.rolling(params.flow_window, min_periods=params.flow_window).sum()
        out['return_sum'] = out['return'].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out['return_volume_sum'] = out['return_volume'].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out['weighted_return'] = out['return_volume_sum'] / out['volume_sum'].where(out['volume_sum'] > 0.0)
        out['participation_gap'] = out['return_sum'] - params.flow_window * out['weighted_return']
        out['rank'] = out['participation_gap'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        rank = indicators['rank']
        gap = indicators['participation_gap']
        drift = indicators['return_sum']
        long_setup = (rank <= params.tail) & (gap < 0.0) & (drift < 0.0)
        short_setup = (rank >= 1.0 - params.tail) & (gap > 0.0) & (drift > 0.0)
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if side != 0:
                # The preceding close's decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                        continue
                age = i - entry_bar + 1
                gain = side * (closes[i] / entry_price - 1.0)
                if age >= params.max_hold or (np.isfinite(gain) and gain >= params.profit_target):
                    side = 0
                    # Exit decisions also execute at the following open.
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                if long_entry[i]:
                    side = 1
                elif short_entry[i]:
                    side = -1
                if side != 0:
                    entry_bar = i + 1
                    entry_price = np.nan
            signals[i] = side

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': [100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'rank': [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5], 'participation_gap': [-0.01, -0.01, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0], 'return_sum': [-0.02, -0.02, 0.0, 0.0, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0]}, index=index)
    strategy = GeneratedStrategy()
    params = ParticipationParams(max_hold=2)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0, -1, -1, 0, 0]
    data.loc[index[6], 'close'] = 97.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0, -1, 0, 0, 0]
    assert (result['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
