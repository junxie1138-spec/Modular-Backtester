from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    anchor_bars: int = 30
    fast_atr_bars: int = 3
    slow_atr_bars: int = 20
    shock_ratio: float = 1.15
    depth_atr: float = 1.5
    rebound_atr: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791320973'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.anchor_bars + 1, params.fast_atr_bars + 2, params.slow_atr_bars + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.anchor_bars, params.fast_atr_bars, params.slow_atr_bars) < 1 or not 3 <= params.hold_bars <= 5:
            raise ValueError('Lookbacks must be positive and hold_bars must be 3, 4, or 5.')
        if not all(np.isfinite(x) and x > 0 for x in (params.shock_ratio, params.depth_atr, params.rebound_atr)):
            raise ValueError('Thresholds must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            (data['high'] - data['low']).abs(),
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['fast_atr'] = out['true_range'].rolling(params.fast_atr_bars, min_periods=params.fast_atr_bars).mean()
        out['slow_atr'] = out['true_range'].rolling(params.slow_atr_bars, min_periods=params.slow_atr_bars).mean().replace(0.0, np.nan)
        out['peak'] = close.shift(1).rolling(params.anchor_bars, min_periods=params.anchor_bars).max()
        out['trough'] = close.shift(1).rolling(params.anchor_bars, min_periods=params.anchor_bars).min()
        out['drawdown_atr'] = (out['peak'] - close) / out['slow_atr']
        out['rebound_atr'] = (close - out['trough']) / out['slow_atr']
        out['volatility_ratio'] = out['fast_atr'] / out['slow_atr']
        out['atr_change'] = out['fast_atr'].diff()
        out['close_change'] = close.diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['eligible'] = (
            (out['drawdown_atr'] >= params.depth_atr)
            & (out['rebound_atr'] >= params.rebound_atr)
            & (out['volatility_ratio'] >= params.shock_ratio)
        ).fillna(False)
        out['long_setup'] = (
            out['eligible'] & (out['atr_change'] < 0) & (out['close_change'] > 0)
        ).fillna(False)
        out['short_setup'] = (
            out['eligible'] & (out['atr_change'] > 0) & (out['close_change'] < 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be 3, 4, or 5.')
        long_setup = indicators['long_setup'].fillna(False).astype(bool)
        short_setup = indicators['short_setup'].fillna(False).astype(bool)
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=14)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    setups = pd.DataFrame({
        'long_setup': [True, True] + [False] * 12,
        'short_setup': [False, False, False, True, True, False, False, True, True, False, False, False, False, False],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, setups, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0]
    assert (result['size'] > 0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat['signal'].eq(0).all()
