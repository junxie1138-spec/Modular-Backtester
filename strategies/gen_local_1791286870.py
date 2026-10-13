from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    min_streak: int = 3
    min_body: float = 0.001
    profit_target: float = 0.006
    max_hold: int = 2

    def __post_init__(self):
        if self.min_streak < 2:
            raise ValueError('min_streak must be at least 2')
        if not np.isfinite(self.min_body) or self.min_body < 0:
            raise ValueError('min_body must be finite and nonnegative')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if self.max_hold not in (1, 2):
            raise ValueError('max_hold must be 1 or 2')


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = 'gen_local_1791286870'

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    @staticmethod
    def warmup_bars(params: AbsorptionParams) -> int:
        return int(params.min_streak + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        close = data['close']
        opening = data['open']
        valid = np.isfinite(close) & np.isfinite(opening) & close.gt(0) & opening.gt(0)
        change = close.diff()
        direction = np.sign(change).where(valid & valid.shift(1, fill_value=False), 0).fillna(0).astype(int)
        groups = direction.ne(direction.shift(1)).cumsum()
        length = (direction.groupby(groups).cumcount() + 1).clip(upper=params.min_streak + 1)
        streak = (length * direction).astype(int)
        body = (close / opening.where(opening.gt(0)) - 1).replace([np.inf, -np.inf], np.nan)
        buy_absorption = valid & body.gt(params.min_body)
        sell_absorption = valid & body.lt(-params.min_body)
        long_entry = (
            streak.le(-params.min_streak)
            & streak.shift(1).le(-params.min_streak)
            & buy_absorption
            & buy_absorption.shift(1, fill_value=False)
        )
        short_entry = (
            streak.ge(params.min_streak)
            & streak.shift(1).ge(params.min_streak)
            & sell_absorption
            & sell_absorption.shift(1, fill_value=False)
        )
        return pd.DataFrame({
            'valid': valid,
            'direction': direction,
            'streak': streak,
            'body': body,
            'buy_absorption': buy_absorption,
            'sell_absorption': sell_absorption,
            'entry': np.select([long_entry, short_entry], [1, -1], default=0).astype(int),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AbsorptionParams) -> SignalFrame:
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        fill_bar = -1
        for i in range(len(data)):
            if position:
                if i == fill_bar:
                    entry_price = opening[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        raise ValueError('Entry open must be finite and positive')
                age = i - fill_bar + 1
                gain = position * (closing[i] / entry_price - 1)
                # Closing targets are evaluated now; exits fill at the next open.
                if (np.isfinite(gain) and gain >= params.profit_target) or age >= params.max_hold:
                    position = 0
                    continue
                signals[i] = position
            elif entries[i]:
                position = int(entries[i])
                fill_bar = i + 1
                signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    opening = np.array([100, 100, 99, 96, 95, 94, 95, 94], dtype=float)
    closing = np.array([100, 99, 98, 97, 96, 95, 94, 93], dtype=float)
    data = pd.DataFrame({
        'open': opening,
        'high': np.maximum(opening, closing) + 1,
        'low': np.minimum(opening, closing) - 1,
        'close': closing,
        'volume': 1000.0,
    }, index=pd.date_range('2020-01-01', periods=8))
    strategy = GeneratedStrategy
    params = AbsorptionParams(profit_target=1.0)
    derived = strategy.indicators(data, params)
    assert derived['entry'].tolist()[:5] == [0, 0, 0, 0, 1]
    result = strategy.generate_signals(data, derived, None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 0]
    params = AbsorptionParams()
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    mirrored = data.copy()
    mirrored['open'] = 200 - data['open']
    mirrored['close'] = 200 - data['close']
    mirrored['high'] = 200 - data['low']
    mirrored['low'] = 200 - data['high']
    params = AbsorptionParams(profit_target=1.0)
    result = strategy.generate_signals(mirrored, strategy.indicators(mirrored, params), None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 0, 0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
