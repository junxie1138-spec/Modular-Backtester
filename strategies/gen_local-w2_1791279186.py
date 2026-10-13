from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionAccelerationParams:
    range_bars: int = 6
    baseline_bars: int = 30
    roc_bars: int = 3
    noise_bars: int = 14
    atr_bars: int = 14
    compression_limit: float = 0.75
    acceleration_threshold: float = 0.60
    efficiency_threshold: float = 0.35
    stop_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[CompressionAccelerationParams]):
    strategy_id = 'gen_local-w2_1791279186'

    @classmethod
    def params_type(cls) -> type[CompressionAccelerationParams]:
        return CompressionAccelerationParams

    def warmup_bars(self, params: CompressionAccelerationParams) -> int:
        return int(max(params.range_bars + params.baseline_bars,
                       2 * params.roc_bars + 1,
                       params.noise_bars + 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: CompressionAccelerationParams) -> pd.DataFrame:
        p = params
        if min(p.range_bars, p.baseline_bars, p.roc_bars, p.noise_bars, p.atr_bars, p.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0 < p.compression_limit < 1 and p.acceleration_threshold > 0
                and 0 <= p.efficiency_threshold <= 1 and p.stop_atr > 0
                and 0 < p.min_size <= p.max_size <= 1):
            raise ValueError('Invalid thresholds or position sizes.')
        close = data['close'].where(data['close'] > 0)
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out['return'] = close.pct_change(fill_method=None)
        out['roc'] = close.pct_change(p.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff(p.roc_bars)
        out['noise'] = out['return'].abs().rolling(p.noise_bars).mean()
        out['path'] = out['return'].abs().rolling(p.roc_bars).sum()
        out['efficiency'] = out['roc'].abs().div(out['path'].replace(0, np.nan)).clip(0, 1)
        out['envelope'] = (data['high'].rolling(p.range_bars).max()
                           - data['low'].rolling(p.range_bars).min()).div(close)
        out['baseline'] = out['envelope'].shift(1).rolling(p.baseline_bars).mean()
        out['compression'] = out['envelope'].div(out['baseline'].replace(0, np.nan))
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_bars).mean()
        out['strength'] = out['acceleration'].abs().div(
            (p.roc_bars * out['noise']).replace(0, np.nan))
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out.notna().all(axis=1) & close.notna() & (out['atr'] > 0)
        eligible = (out['ready']
                    & (out['compression'] <= p.compression_limit)
                    & (out['strength'] >= p.acceleration_threshold)
                    & (out['efficiency'] >= p.efficiency_threshold)
                    & (out['roc'] * out['acceleration'] > 0))
        out['entry_side'] = np.where(eligible, np.sign(out['acceleration']), 0)
        return out.fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionAccelerationParams) -> SignalFrame:
        p = params
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, p.min_size, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry_side'].to_numpy(dtype=int)
        strength = indicators['strength'].to_numpy(dtype=float)
        ready = indicators['ready'].to_numpy(dtype=bool)
        side = 0
        entry_bar = -1
        stop = np.nan
        position_size = p.min_size
        for i in range(n):
            if side:
                if i == entry_bar:
                    # The prior close's decision fills at this bar's open.
                    stop = opening[i] - side * p.stop_atr * atr[i - 1]
                invalid = (not np.isfinite(close[i]) or close[i] <= 0
                           or not np.isfinite(stop)
                           or (i == entry_bar and opening[i] <= 0))
                stopped = side * (close[i] - stop) < 0
                expired = i - entry_bar + 1 >= p.hold_bars
                if invalid or stopped or expired:
                    side = 0
                    continue
            elif ready[i] and candidates[i] and np.isfinite(close[i]) and close[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
                position_size = float(np.clip(
                    p.min_size * strength[i] / p.acceleration_threshold,
                    p.min_size, p.max_size))
            signals[i] = side
            if side:
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Both stops use the entry open and the decision bar's frozen ATR.
    index = pd.date_range('2020-01-01', periods=7)
    sample = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [104.0] * 7,
        'low': [96.0] * 7,
        'close': [100.0, 101.0, 97.0, 100.0, 99.0, 103.0, 100.0],
        'volume': [1000.0] * 7,
    }, index=index)
    features = pd.DataFrame({
        'atr': [1.0, 1.0, 10.0, 1.0, 1.0, 10.0, 1.0],
        'entry_side': [1, 0, 0, -1, 0, 0, 0],
        'strength': [0.96, 0.0, 0.0, 1.92, 0.0, 0.0, 0.0],
        'ready': [True] * 7,
    }, index=index)
    params = CompressionAccelerationParams()
    result = GeneratedStrategy.generate_signals(None, sample, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result['size'].iloc[[1, 4]], [0.4, 0.8])
    assert (result['size'] > 0).all()
    features['entry_side'] = [1, 0, 0, 0, 0, 0, 0]
    sample['close'] = 100.0
    params.hold_bars = 2
    result = GeneratedStrategy.generate_signals(None, sample, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
