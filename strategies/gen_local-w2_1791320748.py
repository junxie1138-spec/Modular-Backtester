from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    acceleration_threshold: float = 0.25
    min_drawdown_atr: float = 1.5
    max_drawdown_atr: float = 7.0
    trail_atr: float = 2.0
    hold_bars: int = 4
    min_size: float = 0.15
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791320748'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1, params.roc_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.roc_window, params.atr_window) < 1:
            raise ValueError('Indicator windows must be positive')
        if not (0 < params.acceleration_threshold and 0 < params.trail_atr):
            raise ValueError('Acceleration threshold and ATR multiplier must be positive')
        if not (0 < params.min_drawdown_atr < params.max_drawdown_atr):
            raise ValueError('Drawdown bounds must be positive and ordered')
        if not (0 < params.min_size <= params.max_size <= 1 and 3 <= params.hold_bars <= 5):
            raise ValueError('Invalid sizing bounds or holding horizon')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean().where(lambda s: s > 0)
        out['atr_fraction'] = out['atr'] / close
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown_atr'] = (out['peak'] - close) / out['atr']
        out['roc'] = close.pct_change(params.roc_window, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['normalized_acceleration'] = out['acceleration'] / out['atr_fraction']
        out['previous_acceleration'] = out['normalized_acceleration'].shift(1)
        active = out['drawdown_atr'].between(params.min_drawdown_atr, params.max_drawdown_atr)
        long_entry = active & (out['roc'] < 0) & (out['normalized_acceleration'] >= params.acceleration_threshold) & (out['previous_acceleration'] < params.acceleration_threshold)
        short_entry = active & (out['roc'] > 0) & (out['normalized_acceleration'] <= -params.acceleration_threshold) & (out['previous_acceleration'] > -params.acceleration_threshold)
        out['valid'] = np.isfinite(out[['atr', 'drawdown_atr', 'roc', 'normalized_acceleration', 'previous_acceleration']]).all(axis=1) & close.notna()
        out.iloc[:self.warmup_bars(params) - 1, out.columns.get_loc('valid')] = False
        out['entry'] = np.select([long_entry & out['valid'], short_entry & out['valid']], [1, -1], default=0).astype(int)
        out['entry_size'] = (params.max_size * out['normalized_acceleration'].abs() / (2.0 * params.acceleration_threshold)).clip(params.min_size, params.max_size).fillna(params.min_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        strengths = indicators['entry_size'].to_numpy(dtype=float)
        valid = indicators['valid'].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = 0
        extreme = 0.0
        stop = 0.0
        held_size = params.min_size
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not valid[i] or not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                continue
            if position:
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the high-water trailing stop for short exposure.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i
                extreme = close
                stop = close - position * params.trail_atr * atr
                held_size = float(np.clip(strengths[i], params.min_size, params.max_size))
            signals[i] = position
            if position:
                sizes[i] = held_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=6)
    ind = pd.DataFrame({'atr': 1.0, 'entry': [1, 0, 0, 0, 0, 0], 'entry_size': 0.4, 'valid': True}, index=index)
    prices = pd.DataFrame({'close': [100., 102., 101., 98., 98., 98.]}, index=index)
    result = GeneratedStrategy.generate_signals(None, prices, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result['size'].iloc[1] == 0.4
    ind['entry'] = [-1, 0, 0, 0, 0, 0]
    prices['close'] = [100., 98., 99., 102., 102., 102.]
    result = GeneratedStrategy.generate_signals(None, prices, ind, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, 0, 0]
    ind['entry'] = [1, 0, 0, 0, 0, 0]
    prices['close'] = 100.
    result = GeneratedStrategy.generate_signals(None, prices, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
