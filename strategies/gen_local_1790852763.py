from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    streak_bars: int = 3
    flow_bars: int = 3
    hold_bars: int = 4

    def __post_init__(self):
        if self.streak_bars < 2 or self.flow_bars < 2:
            raise ValueError('Streak and flow windows must be at least two bars.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('Holding period must be three to five bars.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1790852763'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(params.streak_bars, params.flow_bars) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & data['close'].gt(0))
        volume = data['volume'].where(np.isfinite(data['volume']) & data['volume'].ge(0))
        out['change'] = close.diff()
        falling = out['change'].lt(0)
        rising = out['change'].gt(0)
        out['down_streak'] = falling.astype(int).groupby((~falling).cumsum()).cumsum().clip(upper=params.streak_bars)
        out['up_streak'] = rising.astype(int).groupby((~rising).cumsum()).cumsum().clip(upper=params.streak_bars)
        out['signed_volume'] = np.sign(out['change']) * volume
        out['net_flow'] = out['signed_volume'].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        return out

    def generate_signals(self, data, indicators, ctx, params):
        long_setup = (
            indicators['down_streak'].shift(1).ge(params.streak_bars)
            & indicators['change'].gt(0)
            & indicators['net_flow'].gt(0)
        )
        short_setup = (
            indicators['up_streak'].shift(1).ge(params.streak_bars)
            & indicators['change'].lt(0)
            & indicators['net_flow'].lt(0)
        )
        entries = np.select([long_setup.fillna(False), short_setup.fillna(False)], [1, -1], default=0)
        entries[:self.warmup_bars(params) - 1] = 0
        positions = np.zeros(len(data), dtype=int)
        position = 0
        deadline = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == deadline:
                    position = 0
                positions[i] = position
                continue
            if entry != 0:
                position = int(entry)
                deadline = i + params.hold_bars
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=12)
    prices = np.array([100, 99, 98, 97, 98, 99, 98, 99, 100, 101, 100, 99], dtype=float)
    for direction in (1, -1):
        close = prices if direction == 1 else 200.0 - prices
        data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': [1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1]}, index=index)
        indicators = strategy.indicators(data, params)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0] * 5 + [direction] * 4 + [0] * 3
        assert result['size'].gt(0).all()
