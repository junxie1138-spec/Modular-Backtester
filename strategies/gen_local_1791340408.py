from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    ma_window: int = 20
    compression_window: int = 5
    percentile_window: int = 126
    compression_percentile: float = 0.25
    stretch_percentile: float = 0.80
    atr_window: int = 14
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local_1791340408'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.ma_window, params.compression_window, params.atr_window) + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.ma_window, params.compression_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not (0 < params.compression_percentile < 1 and 0 < params.stretch_percentile < 1):
            raise ValueError('Percentiles must lie strictly between zero and one.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or not 3 <= params.holding_bars <= 5:
            raise ValueError('Use a positive ATR multiplier and a three-to-five-bar holding period.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0).replace(0, np.nan)
        out['z'] = (close - out['ma']) / out['std']
        out['abs_z'] = out['z'].abs()
        out['stretch_threshold'] = out['abs_z'].rolling(params.percentile_window).quantile(params.stretch_percentile).shift(1)
        out['normalized_range'] = (data['high'] - data['low']).clip(lower=0) / close
        out['compression'] = out['normalized_range'].rolling(params.compression_window).mean()
        out['compression_threshold'] = out['compression'].rolling(params.percentile_window).quantile(params.compression_percentile).shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean().replace(0, np.nan)
        out['z_change'] = out['z'].diff()
        out['previous_z'] = out['z'].shift(1)
        out['previous_abs_z'] = out['abs_z'].shift(1)
        out['previous_stretch_threshold'] = out['stretch_threshold'].shift(1)
        out['previous_compression'] = out['compression'].shift(1)
        out['previous_compression_threshold'] = out['compression_threshold'].shift(1)

        valid = np.isfinite(out).all(axis=1)
        compressed = out['previous_compression'] <= out['previous_compression_threshold']
        stretched = out['previous_abs_z'] > out['previous_stretch_threshold']
        inward = out['previous_z'] * out['z_change'] < 0
        same_side = out['previous_z'] * out['z'] > 0
        setup = valid & compressed & stretched & inward & same_side
        out['entry'] = np.select([setup & (out['z'] < 0), setup & (out['z'] > 0)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = low_water = stop = np.nan

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0):
                position = 0
                signals[i] = 0
                continue

            if position:
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.holding_bars:
                    position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i
                high_water = low_water = close
                stop = close - position * params.trail_atr * atr

            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=15)
    data = pd.DataFrame({'close': [100, 100, 102, 101, 99, 100, 98, 100, 101, 102, 103, 104, 105, 106, 107]}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'entry': [1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0]}, index=index)
    params = CompressionParams(holding_bars=5)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 1, 0]
    assert (result.data['size'] > 0).all()
    flat = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=200))
    derived = GeneratedStrategy.indicators(None, flat, params)
    assert derived['entry'].eq(0).all()


if __name__ == '__main__':
    _self_check()
