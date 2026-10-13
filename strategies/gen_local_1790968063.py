from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_days: int = 3
    smoothing: int = 2
    threshold: float = 0.10

    def __post_init__(self):
        if self.roc_days < 1 or self.smoothing < 1:
            raise ValueError('Lookbacks must be positive.')
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('Threshold must be finite and positive.')


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = 'gen_local_1790968063'

    @classmethod
    def params_type(cls):
        return AccelerationParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, 21, params.roc_days + params.smoothing + 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        opening = data['open'].astype(float).where(data['open'] > 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return'] = np.log(close / close.shift(1))
        out['volatility'] = out['return'].rolling(20, min_periods=20).std(ddof=0)
        out['session_return'] = np.log(close / opening)
        out['overnight_return'] = np.log(opening / close.shift(1))
        n = params.roc_days
        s = params.smoothing
        out['roc'] = out['return'].rolling(n, min_periods=n).sum()
        out['session_roc'] = out['session_return'].rolling(n, min_periods=n).sum()
        out['overnight_roc'] = out['overnight_return'].rolling(n, min_periods=n).sum()
        out['acceleration'] = out['roc'].diff().rolling(s, min_periods=s).mean()
        out['session_acceleration'] = out['session_roc'].diff().rolling(s, min_periods=s).mean()
        out['overnight_acceleration'] = out['overnight_roc'].diff().rolling(s, min_periods=s).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['cutoff'] = params.threshold * out['volatility']
        valid = out.notna().all(axis=1) & out['volatility'].gt(0)
        bullish = valid & out['acceleration'].gt(out['cutoff']) & out['session_acceleration'].gt(0) & out['overnight_acceleration'].le(0)
        bearish = valid & out['acceleration'].lt(-out['cutoff']) & out['session_acceleration'].lt(0) & out['overnight_acceleration'].ge(0)
        out['direction'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        out['regime'] = np.select([close.gt(out['ma200']), close.lt(out['ma200'])], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        positions = np.zeros(len(data), dtype=int)
        position = 0
        directions = indicators['direction'].fillna(0).to_numpy(dtype=int)
        regimes = indicators['regime'].fillna(0).to_numpy(dtype=int)
        for i, (direction, regime) in enumerate(zip(directions, regimes)):
            if position != 0:
                # Only an opposite qualifying acceleration exits; neutral bars hold.
                # The MA gates new entries, not exits from existing positions.
                if direction == -position:
                    position = 0
            elif direction != 0 and direction == regime:
                position = int(direction)
            positions[i] = position
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'direction': [1, 0, -1, -1, 1, 1], 'regime': [1, 1, 1, -1, -1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, AccelerationParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
