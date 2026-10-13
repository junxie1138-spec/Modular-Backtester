from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 24
    persistence_window: int = 12
    scale_window: int = 20
    compression_window: int = 5
    baseline_window: int = 40
    compression_ratio: float = 0.80
    persistence_threshold: float = 0.15
    innovation_threshold: float = 0.50
    atr_window: int = 14
    stop_atr: float = 1.50
    hold_days: int = 4

    def __post_init__(self):
        windows = (self.fractional_window, self.persistence_window, self.scale_window, self.compression_window, self.baseline_window, self.atr_window)
        if any(not isinstance(w, int) or w < 2 for w in windows):
            raise ValueError('Lookback windows must be integers >= 2')
        if not 0 < self.fractional_order < 1:
            raise ValueError('fractional_order must be between zero and one')
        if not 0 < self.compression_ratio < 1:
            raise ValueError('compression_ratio must be between zero and one')
        if not 0 <= self.persistence_threshold < 1:
            raise ValueError('persistence_threshold must be in [0, 1)')
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0:
            raise ValueError('innovation_threshold must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not isinstance(self.hold_days, int) or not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be an integer from three to five')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791194712'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.fractional_window + max(params.persistence_window + 5, params.scale_window + 1, params.baseline_window + 1, params.compression_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = [1.0]
        for lag in range(1, params.fractional_window):
            weights.append(-weights[-1] * (params.fractional_order - lag + 1) / lag)
        # ponytail: finite fractional-memory kernel; increase fractional_window for longer memory.
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        out['innovation'] = fractional.diff()
        out['innovation_scale'] = out['innovation'].rolling(params.scale_window, min_periods=params.scale_window).std(ddof=0)
        out['innovation_z'] = out['innovation'] / out['innovation_scale'].replace(0.0, np.nan)
        out['persistence'] = out['innovation'].rolling(params.persistence_window, min_periods=params.persistence_window).corr(out['innovation'].shift(1)).clip(-1.0, 1.0)
        out['recent_negative_persistence'] = out['persistence'].shift(1).rolling(4, min_periods=4).min()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([(data['high'] - data['low']).abs(), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['short_range'] = out['true_range'].rolling(params.compression_window, min_periods=params.compression_window).mean()
        out['baseline_range'] = out['true_range'].shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out['compression'] = out['short_range'] / out['baseline_range'].replace(0.0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        transition = (out['persistence'] > params.persistence_threshold) & (out['persistence'].shift(1) <= params.persistence_threshold) & (out['recent_negative_persistence'] < -0.15)
        eligible = transition & (out['compression'] < params.compression_ratio) & (out['innovation_z'].abs() >= params.innovation_threshold) & (out['atr'] > 0) & close.notna()
        out['entry_direction'] = np.where(eligible.fillna(False), np.sign(out['innovation_z'].fillna(0.0)), 0).astype(int)
        magnitude = out['innovation_z'].abs()
        out['conviction'] = ((magnitude / (1.0 + magnitude)) * ((1.0 + out['persistence']) / 2.0)).fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        count = len(data)
        signal = np.zeros(count, dtype=int)
        size = np.full(count, 0.15, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        direction = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        conviction = indicators['conviction'].fillna(0.0).to_numpy(dtype=float)
        position = 0
        pending = 0
        pending_atr = 0.0
        entry_bar = 0
        stop = 0.0
        exposure = 0.15
        for i in range(count):
            # A previous close's entry decision becomes a position at this open.
            if pending:
                position = pending if np.isfinite(opening[i]) and opening[i] > 0 else 0
                pending = 0
                entry_bar = i
                stop = opening[i] - position * params.stop_atr * pending_atr
            if position:
                stopped = not np.isfinite(close[i]) or position * (close[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    position = 0
                else:
                    signal[i] = position
                    size[i] = exposure
            elif direction[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]) and close[i] > 0:
                pending = int(direction[i])
                pending_atr = float(atr[i])
                exposure = float(0.15 + 0.80 * np.clip(conviction[i], 0.0, 1.0))
                signal[i] = pending
                size[i] = exposure
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=9, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': [100, 100, 103, 97, 100, 100, 97, 103, 100], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'entry_direction': [1, 0, 0, 0, -1, 0, 0, 0, 0], 'conviction': [0.25, 0, 0, 0, 0.75, 0, 0, 0, 0]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert np.isclose(result['size'].iloc[1], 0.35) and np.isclose(result['size'].iloc[5], 0.75)
    data['close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    derived = strategy.indicators(data, Params())
    assert derived['entry_direction'].eq(0).all()
    assert derived['conviction'].eq(0.0).all()


if __name__ == '__main__':
    _self_check()
