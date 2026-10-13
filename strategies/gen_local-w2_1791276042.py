from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StarvationParams:
    streak_length: int = 3
    stop_atr: float = 2.0

    def __post_init__(self):
        if self.streak_length < 1:
            raise ValueError('streak_length must be positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[StarvationParams]):
    strategy_id = 'gen_local-w2_1791276042'

    @classmethod
    def params_type(cls) -> type[StarvationParams]:
        return StarvationParams

    @staticmethod
    def warmup_bars(params: StarvationParams) -> int:
        return int(max(15, params.streak_length + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StarvationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = data['high'] - data['low']
        out['previous_close'] = data['close'].shift(1)
        out['previous_high'] = data['high'].shift(1)
        out['previous_low'] = data['low'].shift(1)
        out['previous_range'] = out['range'].shift(1)
        out['previous_volume'] = data['volume'].shift(1)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['compression'] = (
            (out['range'] > 0)
            & (out['range'] < out['previous_range'])
            & (data['volume'] > 0)
            & (data['volume'] < out['previous_volume'])
        )
        compression = out['compression']
        out['streak'] = compression.astype(int).groupby((~compression).cumsum()).cumsum()
        out['previous_streak'] = out['streak'].shift(1).fillna(0)
        out['release'] = (
            (out['previous_streak'] >= params.streak_length)
            & (out['range'] > out['previous_range'])
            & (data['volume'] > out['previous_volume'])
            & (out['atr'] > 0)
        )
        out['long_entry'] = out['release'] & (data['close'] > out['previous_high'])
        out['short_entry'] = out['release'] & (data['close'] < out['previous_low'])
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StarvationParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue

            if position:
                age += 1
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.stop_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.stop_atr * atr)
                    stopped = close >= stop
                if stopped or age >= 4:
                    position = 0
                    age = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                extreme = close
                stop = close - position * params.stop_atr * atr

            decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100, 103, 102, 100, 100, 99, 96, 98]}, index=index)
    derived = pd.DataFrame({
        'atr': 1.0,
        'long_entry': [True, False, False, False, False, False, False, False],
        'short_entry': [False, False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, StarvationParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1]
    data['close'] = 100.0
    derived['short_entry'] = False
    result = GeneratedStrategy.generate_signals(data, derived, None, StarvationParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
