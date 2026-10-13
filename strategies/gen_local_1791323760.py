from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeShiftParams:
    recent_window: int = 5
    baseline_window: int = 40
    confirmation_bars: int = 2
    breakout_window: int = 20
    atr_window: int = 14
    shift_threshold: float = 1.25
    max_range_cv: float = 0.45
    stop_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.35
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.recent_window, self.baseline_window, self.breakout_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two bars')
        if self.confirmation_bars < 1 or self.hold_bars < 1:
            raise ValueError('Confirmation and holding periods must be positive')
        if not all(np.isfinite(x) and x > 0 for x in (self.shift_threshold, self.max_range_cv, self.stop_atr)):
            raise ValueError('Thresholds must be finite and positive')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Position sizes must satisfy 0 < min_size <= max_size <= 1')


class GeneratedStrategy(BaseStrategy[RangeShiftParams]):
    strategy_id = 'gen_local_1791323760'

    @classmethod
    def params_type(cls):
        return RangeShiftParams

    def warmup_bars(self, params: RangeShiftParams) -> int:
        return int(max(
            params.recent_window + params.baseline_window + params.confirmation_bars + 1,
            params.breakout_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: RangeShiftParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda x: x > 0)
        out['normalized_range'] = ((data['high'] - data['low']) / previous_close).replace([np.inf, -np.inf], np.nan)
        ranges = out['normalized_range'].where(out['normalized_range'] >= 0)
        out['recent_mean'] = ranges.rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['recent_std'] = ranges.rolling(params.recent_window, min_periods=params.recent_window).std(ddof=0)
        historical = ranges.shift(params.recent_window)
        out['baseline_mean'] = historical.rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out['baseline_std'] = historical.rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        scale = out['baseline_std'].clip(lower=1e-6)
        out['shift_strength'] = ((out['recent_mean'] - out['baseline_mean']) / scale).replace([np.inf, -np.inf], np.nan)
        out['range_cv'] = (out['recent_std'] / out['recent_mean'].where(out['recent_mean'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['expanded'] = ((out['shift_strength'] >= params.shift_threshold) & (out['range_cv'] <= params.max_range_cv)).astype(int)
        out['confirmed'] = out['expanded'].rolling(params.confirmation_bars, min_periods=params.confirmation_bars).sum().eq(params.confirmation_bars)
        out['new_regime'] = out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        out['breakout_level'] = data['high'].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).max()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['entry'] = (out['new_regime'] & data['close'].gt(out['breakout_level']) & data['close'].gt(data['open']) & out['atr'].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeShiftParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        strength = indicators['shift_strength'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        stop_price = np.nan
        position_size = params.min_size
        held = 0

        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    held = 0
                    # Entry fills at this open; ATR was known at the preceding close.
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                else:
                    continue

            if active:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or held >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i] and np.isfinite(strength[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                fraction = float(np.clip(strength[i] / (2.0 * params.shift_threshold), 0.0, 1.0))
                position_size = params.min_size + (params.max_size - params.min_size) * fraction
                pending = True
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0, 110.0, 109.0, 105.0, 105.0],
        'high': [101.0, 112.0, 110.0, 106.0, 106.0],
        'low': [99.0, 109.0, 104.0, 104.0, 104.0],
        'close': [100.0, 111.0, 105.0, 105.0, 105.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2.0] * 5,
        'shift_strength': [2.0] * 5,
        'entry': [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = RangeShiftParams(stop_atr=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    assert params.min_size < result['size'].iloc[1] < params.max_size
    assert result['size'].gt(0).all()
    # A later ATR increase must not lower the fixed entry stop.
    indicators.loc[index[2]:, 'atr'] = 100.0
    fixed = strategy.generate_signals(data, indicators, None, params).data
    assert fixed['signal'].equals(result['signal'])
    params = RangeShiftParams(stop_atr=100.0, hold_bars=2)
    timed = strategy.generate_signals(data, indicators, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0]
    warm = strategy.indicators(data, RangeShiftParams())
    assert warm.index.equals(data.index) and not warm['entry'].any()


if __name__ == '__main__':
    _self_check()
