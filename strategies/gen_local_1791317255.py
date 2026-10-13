from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 48
    channel_window: int = 20
    breakout_buffer: float = 0.15
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791317255'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.fractional_window + params.channel_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError('fractional_order must be between zero and one')
        if params.fractional_window < 2 or params.channel_window < 2:
            raise ValueError('Indicator windows must be at least two')
        if params.hold_bars < 1 or not (0.0 <= params.breakout_buffer < np.inf):
            raise ValueError('Invalid holding period or breakout buffer')
        if not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')

        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        fractional_close = close.copy()
        weight = 1.0
        # Fixed-width fractional differencing: newest close has weight one.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fractional_close
        prior = fractional_close.shift(1)
        out['upper'] = prior.rolling(params.channel_window, min_periods=params.channel_window).max()
        out['lower'] = prior.rolling(params.channel_window, min_periods=params.channel_window).min()
        out['scale'] = fractional_close.diff().shift(1).rolling(
            params.channel_window, min_periods=params.channel_window
        ).std(ddof=0).replace(0.0, np.nan)
        out['upper_trigger'] = out['upper'] + params.breakout_buffer * out['scale']
        out['lower_trigger'] = out['lower'] - params.breakout_buffer * out['scale']

        upward = (fractional_close > out['upper_trigger']) & (
            fractional_close.shift(1) <= out['upper_trigger'].shift(1)
        )
        downward = (fractional_close < out['lower_trigger']) & (
            fractional_close.shift(1) >= out['lower_trigger'].shift(1)
        )
        ready = out.notna().all(axis=1)
        out['candidate'] = np.select(
            [ready & upward, ready & downward], [1, -1], default=0
        ).astype(int)
        out['overshoot'] = pd.concat([
            (fractional_close - out['upper_trigger']) / out['scale'],
            (out['lower_trigger'] - fractional_close) / out['scale'],
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        out['conviction'] = (
            params.min_size + (params.max_size - params.min_size)
            * out['overshoot'] / (1.0 + out['overshoot'])
        ).clip(params.min_size, params.max_size)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        candidate = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        conviction = indicators['conviction'].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(params.min_size).clip(params.min_size, params.max_size).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        position, entry, entry_size = 0, -1, params.min_size

        for i in range(len(data)):
            if position:
                if i - entry >= params.hold_bars:
                    position = 0
                else:
                    signal[i], size[i] = position, entry_size
                # Exit bars remain flat; opposite signals cannot shorten a hold.
                continue
            if candidate[i]:
                position, entry, entry_size = int(candidate[i]), i, float(conviction[i])
                signal[i], size[i] = position, entry_size

        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=25, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'candidate': [1] + [-1] * 24,
        'conviction': [0.4] + [0.8] * 24,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:9] == 1).all()
    assert result['signal'].iloc[9] == 0
    assert (result['signal'].iloc[10:18] == -1).all()
    assert result['signal'].iloc[18] == 0
    assert (result['size'].iloc[1:9] == 0.4).all()
    assert (result['size'].iloc[10:18] == 0.8).all()


if __name__ == '__main__':
    _self_check()
