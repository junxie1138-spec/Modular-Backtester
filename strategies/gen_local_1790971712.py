from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalBreakoutParams:
    order: float = 0.45
    memory: int = 64
    breakout_window: int = 30
    hold_bars: int = 18

    def __post_init__(self):
        if not np.isfinite(self.order) or not 0.0 < self.order < 1.0:
            raise ValueError('order must be between zero and one')
        for name in ('memory', 'breakout_window', 'hold_bars'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{name} must be an integer >= 2')


class GeneratedStrategy(BaseStrategy[FractionalBreakoutParams]):
    strategy_id = 'gen_local_1790971712'

    @classmethod
    def params_type(cls):
        return FractionalBreakoutParams

    def warmup_bars(self, params):
        return int(params.memory + params.breakout_window + 1)

    def indicators(self, data, params):
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        fractional_close = close.copy()
        weight = 1.0
        # Fixed-width fractional difference; all weights use past closes only.
        for lag in range(1, params.memory):
            weight *= (lag - 1.0 - params.order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        fractional_close = fractional_close.where(np.isfinite(fractional_close))
        ceiling = fractional_close.rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max().shift(1)
        breakout = (
            fractional_close.notna()
            & ceiling.notna()
            & fractional_close.gt(ceiling)
        )
        confirmed = breakout & breakout.shift(1, fill_value=False)
        return pd.DataFrame({
            'fractional_close': fractional_close,
            'ceiling': ceiling,
            'breakout': breakout,
            'confirmed': confirmed,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                    continue
                signal[bar] = 1
            elif ready[bar]:
                signal[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = FractionalBreakoutParams()
    index = pd.date_range('2020-01-01', periods=30, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'confirmed': False}, index=index)
    indicators.iloc[2, 0] = True
    indicators.iloc[8, 0] = True
    indicators.iloc[20, 0] = True
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, params
    ).data
    expected = np.zeros(len(index), dtype=int)
    expected[3:21] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
