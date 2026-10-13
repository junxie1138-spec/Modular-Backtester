from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    innovation_window: int = 20
    compression_window: int = 5
    baseline_window: int = 20
    compression_ratio: float = 0.75
    impulse_z: float = 1.25
    expansion_ratio: float = 1.1
    atr_window: int = 14
    stop_atr: float = 1.2
    holding_bars: int = 2

    def __post_init__(self):
        windows = (self.fractional_window, self.innovation_window,
                   self.compression_window, self.baseline_window, self.atr_window)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in windows):
            raise ValueError('Lookbacks must be integers of at least two.')
        values = (self.fractional_order, self.compression_ratio,
                  self.impulse_z, self.expansion_ratio, self.stop_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds must be finite and positive.')
        if self.fractional_order >= 1 or self.compression_ratio >= 1:
            raise ValueError('Fractional order and compression ratio must be below one.')
        if isinstance(self.holding_bars, bool) or not isinstance(self.holding_bars, int) or self.holding_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions.')


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791281059'

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(200,
                       params.fractional_window + params.innovation_window + 1,
                       params.compression_window + 1,
                       params.baseline_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(lambda x: x > 0)
        out['log_close'] = np.log(close)
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # Fixed finite-memory fractional filter; every term uses a past or current close.
        fractional = out['log_close'].copy()
        for lag in range(1, params.fractional_window):
            fractional = fractional + weights[lag] * out['log_close'].shift(lag)
        out['fractional_close'] = fractional
        out['innovation'] = fractional.diff()
        reference = out['innovation'].shift(1).rolling(
            params.innovation_window, min_periods=params.innovation_window)
        out['innovation_mean'] = reference.mean()
        out['innovation_std'] = reference.std(ddof=0).replace(0.0, np.nan)
        out['innovation_z'] = ((out['innovation'] - out['innovation_mean']) /
                               out['innovation_std']).replace([np.inf, -np.inf], np.nan)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['recent_range'] = out['true_range'].shift(1).rolling(
            params.compression_window, min_periods=params.compression_window).mean()
        out['baseline_range'] = out['true_range'].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window).mean()
        out['compression'] = (out['recent_range'] /
                              out['baseline_range'].replace(0.0, np.nan))
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['range_release'] = (
            (out['compression'] < params.compression_ratio) &
            (out['true_range'] > params.expansion_ratio * out['recent_range'])
        ).fillna(False)
        ready = np.isfinite(out[['innovation_z', 'atr', 'compression', 'ma200']]).all(axis=1)
        ready &= out['atr'].gt(0) & close.notna()
        long_entry = (ready & out['range_release'] & close.gt(out['ma200']) &
                      out['innovation_z'].gt(params.impulse_z))
        short_entry = (ready & out['range_release'] & close.lt(out['ma200']) &
                       out['innovation_z'].lt(-params.impulse_z))
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = 0.0
        low_water = 0.0
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and close[i] > 0 and
                    np.isfinite(atr[i]) and atr[i] > 0):
                position = 0
                age = 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                # Never loosen the highest-close drawdown barrier, even if ATR expands.
                high_stop = max(high_stop, high_water - params.stop_atr * atr[i])
                short_stop = min(short_stop, low_water + params.stop_atr * atr[i])
                high_exit = close[i] <= high_stop
                # Shorts also receive a mirrored adverse-move trail.
                short_exit = position == -1 and close[i] >= short_stop
                if high_exit or short_exit or age >= params.holding_bars:
                    position = 0
                    age = 0
                signals[i] = position
                # Exit decisions stay flat; do not reopen on the same close.
                continue
            if entries[i]:
                position = int(entries[i])
                age = 0
                # Initialize the trade's reference at its decision close.
                high_water = low_water = close[i]
                high_stop = close[i] - params.stop_atr * atr[i]
                short_stop = close[i] + params.stop_atr * atr[i]
                signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = CompressionParams()
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    prices = np.array([100.0, 97.0, 97.0, 100.0, 103.0, 103.0, 103.0])
    sample = pd.DataFrame({'open': prices, 'high': prices + 1,
                           'low': prices - 1, 'close': prices, 'volume': 1000.0},
                          index=index)
    features = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0, 0], 'atr': 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(None, sample, features, None, params)
    assert result.data['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
    sample['close'] = 100.0
    result = GeneratedStrategy.generate_signals(None, sample, features, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result.data['size'].gt(0).all()
