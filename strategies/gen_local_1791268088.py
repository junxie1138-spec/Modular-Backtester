from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPathParams:
    fractional_order: float = 0.45
    fractional_window: int = 40
    efficiency_window: int = 8
    efficiency_threshold: float = 0.6
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    trailing_atr: float = 3.0
    max_hold_bars: int = 10

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(self.fractional_window, self.efficiency_window, self.volatility_window, self.atr_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if not 0.0 < self.efficiency_threshold <= 1.0:
            raise ValueError('efficiency_threshold must be in (0, 1]')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError('target_volatility must be finite and positive')
        if not np.isfinite(self.trailing_atr) or self.trailing_atr <= 0 or self.max_hold_bars < 1:
            raise ValueError('invalid exit parameters')


class GeneratedStrategy(BaseStrategy[FractionalPathParams]):
    strategy_id = 'gen_local_1791268088'

    @classmethod
    def params_type(cls):
        return FractionalPathParams

    def warmup_bars(self, params: FractionalPathParams) -> int:
        return int(max(200, params.fractional_window + params.efficiency_window,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalPathParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['returns'] = close.pct_change(fill_method=None)
        out['annual_volatility'] = out['returns'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        out['fractional_close'] = fractional_close
        out['fractional_step'] = fractional_close.diff()
        out['fractional_displacement'] = fractional_close.diff(params.efficiency_window)
        out['fractional_path_length'] = out['fractional_step'].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out['efficiency'] = (
            out['fractional_displacement'].abs()
            / out['fractional_path_length'].where(out['fractional_path_length'] > 0)
        ).clip(0.0, 1.0)
        ready = (
            out[['ma200', 'atr', 'annual_volatility', 'efficiency', 'fractional_displacement']]
            .apply(np.isfinite).all(axis=1)
            & (out['atr'] > 0)
            & (out['annual_volatility'] > 0)
        )
        qualified = ready & (out['efficiency'] >= params.efficiency_threshold)
        out['entry_direction'] = np.select([
            qualified & (out['fractional_displacement'] > 0) & (close > out['ma200']),
            qualified & (out['fractional_displacement'] < 0) & (close < out['ma200'])
        ], [1, -1], default=0).astype(int)
        out['target_size'] = (
            params.target_volatility / out['annual_volatility'].where(out['annual_volatility'] > 0)
        ).replace([np.inf, -np.inf], np.nan).clip(0.01, 0.95).fillna(0.01)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FractionalPathParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        closes = data['close'].to_numpy(dtype=float)
        averages = indicators['ma200'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_direction'].to_numpy(dtype=int)
        for i in range(len(data)):
            close, average, atr = closes[i], averages[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(average) and np.isfinite(atr) and atr > 0
            if not valid:
                position = 0
                age = 0
                extreme = stop = np.nan
                continue
            if position:
                age += 1
                if position == 1:
                    extreme = close if np.isnan(extreme) else max(extreme, close)
                    candidate = extreme - params.trailing_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    exit_now = close <= stop or close <= average
                else:
                    extreme = close if np.isnan(extreme) else min(extreme, close)
                    candidate = extreme + params.trailing_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    exit_now = close >= stop or close >= average
                if exit_now or age >= params.max_hold_bars:
                    position = 0
                    age = 0
                    extreme = stop = np.nan
            elif entries[i]:
                position = int(entries[i])
                age = 0
                # The first in-trade close occurs next bar, after the shifted entry.
                extreme = stop = np.nan
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['target_size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = FractionalPathParams()
    index = pd.date_range('2020-01-01', periods=8)
    for direction, closes, average in [
        (1, [100, 102, 105, 110, 106, 105, 104, 103], 90),
        (-1, [110, 108, 105, 100, 104, 105, 106, 107], 200)
    ]:
        data = pd.DataFrame({'close': closes}, index=index)
        indicators = pd.DataFrame({
            'ma200': average, 'atr': 1.0, 'target_size': 0.5,
            'entry_direction': [direction, 0, 0, 0, 0, 0, 0, 0]
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
