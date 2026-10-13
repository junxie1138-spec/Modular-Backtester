from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShiftParams:
    fractional_order: float = 0.4
    fractional_window: int = 40
    recent_window: int = 4
    baseline_window: int = 20
    shift_threshold: float = 2.0
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(self.fractional_window, self.recent_window, self.baseline_window, self.atr_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        if not all(np.isfinite(x) and x > 0 for x in (self.shift_threshold, self.stop_atr)):
            raise ValueError('threshold and stop multiplier must be finite and positive')


class GeneratedStrategy(BaseStrategy[ShiftParams]):
    strategy_id = 'gen_local_1791240324'

    @classmethod
    def params_type(cls):
        return ShiftParams

    @staticmethod
    def warmup_bars(params: ShiftParams) -> int:
        return int(max(200, params.fractional_window + params.baseline_window + params.recent_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShiftParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # ponytail: fixed FIR truncation; lengthen the window if residual memory matters.
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        out['recent_mean'] = fractional.rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['baseline_mean'] = fractional.rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(params.recent_window)
        out['baseline_std'] = fractional.rolling(params.baseline_window, min_periods=params.baseline_window).std().shift(params.recent_window)
        scale = out['baseline_std'].where(out['baseline_std'] > 1e-12) * np.sqrt(1.0 / params.recent_window + 1.0 / params.baseline_window)
        out['shift_score'] = ((out['recent_mean'] - out['baseline_mean']) / scale).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        ranges = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1)
        out['true_range'] = ranges.max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        score = out['shift_score']
        previous_score = score.shift(1)
        valid = close.gt(0) & out['ma200'].notna() & out['atr'].gt(0) & out['atr'].notna()
        bullish = valid & close.gt(out['ma200']) & score.gt(params.shift_threshold) & previous_score.le(params.shift_threshold)
        bearish = valid & close.lt(out['ma200']) & score.lt(-params.shift_threshold) & previous_score.ge(-params.shift_threshold)
        out['candidate'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShiftParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending_entry = False
        entry_price = 0.0
        entry_atr = 0.0
        held = 0
        for i in range(len(data)):
            if position:
                # A decision at the previous close first fills at this bar's open.
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                held += 1
                valid_price = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                # Freeze both entry price and ATR; the stop never trails.
                stopped = valid_price and position * (closes[i] - entry_price) < -params.stop_atr * entry_atr
                if not valid_price or stopped or held >= params.holding_bars:
                    position = 0
                    held = 0
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(candidates[i])
                entry_atr = atrs[i]
                held = 0
                pending_entry = True
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': [100, 105, 100, 100, 100, 105], 'high': [101, 106, 101, 101, 106, 106], 'low': [99, 101, 99, 99, 99, 104], 'close': [100, 102, 100, 100, 105, 105], 'volume': 1000}, index=index)
    indicators = pd.DataFrame({'atr': 2.0, 'candidate': [1, 0, 0, -1, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, ShiftParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, 0]
    assert result['size'].gt(0).all()
    indicators['candidate'] = [1, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(data, indicators, None, ShiftParams(stop_atr=100.0, holding_bars=2)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    derived = GeneratedStrategy.indicators(data, ShiftParams())
    assert derived.index.equals(data.index)
    assert derived['candidate'].eq(0).all()


if __name__ == '__main__':
    _self_check()
