from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 12
    atr_window: int = 14
    area_threshold: float = 0.12
    efficiency_min: float = 0.30
    stop_atr: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        if self.path_window < 2 or self.atr_window < 2 or self.max_hold < 1:
            raise ValueError('Window lengths must be at least two; holding length must be positive.')
        if not 0 < self.area_threshold < 1 or not 0 <= self.efficiency_min <= 1:
            raise ValueError('Invalid area or efficiency threshold.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791312949'

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(params.path_window, params.atr_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_price'] = np.log(data['close'].where(data['close'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['log_volume'] = np.log(data['volume'].where(data['volume'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['dx'] = out['log_price'].diff()
        out['dy'] = out['log_volume'].diff()
        w = params.path_window
        out['cross'] = out['log_price'].shift(1) * out['dy'] - out['log_volume'].shift(1) * out['dx']
        out['price_change'] = out['dx'].rolling(w, min_periods=w).sum()
        out['volume_change'] = out['dy'].rolling(w, min_periods=w).sum()
        out['price_variation'] = out['dx'].abs().rolling(w, min_periods=w).sum()
        out['volume_variation'] = out['dy'].abs().rolling(w, min_periods=w).sum()
        out['area'] = 0.5 * (
            out['cross'].rolling(w, min_periods=w).sum()
            - out['log_price'].shift(w) * out['volume_change']
            + out['log_volume'].shift(w) * out['price_change']
        )
        out['area_scale'] = out['price_variation'] * out['volume_variation']
        # Negative area means volume expansion precedes an upward price move.
        out['score'] = (-2.0 * out['area'] / out['area_scale'].where(out['area_scale'] > 0)).clip(-1.0, 1.0)
        out['efficiency'] = (out['price_change'].abs() / out['price_variation'].where(out['price_variation'] > 0)).clip(0.0, 1.0)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ready'] = np.isfinite(out[['score', 'efficiency', 'price_change', 'atr']]).all(axis=1) & (out['atr'] > 0)
        qualified = out['ready'] & (out['efficiency'] >= params.efficiency_min)
        out['entry'] = np.select([
            qualified & (out['score'] >= params.area_threshold) & (out['price_change'] > 0),
            qualified & (out['score'] <= -params.area_threshold) & (out['price_change'] < 0),
        ], [1, -1], default=0).astype(int)
        out['strength'] = out['score'].abs().fillna(0.0).clip(0.0, 1.0)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        strength = indicators['strength'].to_numpy(dtype=float)
        position = age = 0
        high = low = stop = np.nan
        weight = 1.0
        for i in range(n):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                position = 0
            elif position:
                age += 1
                # Track closes starting on the bar after the entry decision.
                high = max(high, close[i]) if np.isfinite(high) else close[i]
                low = min(low, close[i]) if np.isfinite(low) else close[i]
                if position == 1:
                    stop = max(stop, high - params.stop_atr * atr[i])
                    hit = close[i] <= stop
                else:
                    # Shorts mirror the high-water trail with an in-trade low.
                    stop = min(stop, low + params.stop_atr * atr[i])
                    hit = close[i] >= stop
                if hit or age >= params.max_hold:
                    position = 0
            elif entries[i] != 0 and np.isfinite(strength[i]):
                position = int(entries[i])
                age = 0
                high = low = np.nan
                stop = -np.inf if position == 1 else np.inf
                # Freeze entry conviction to avoid daily resizing churn.
                weight = float(0.20 + 0.80 * np.clip(strength[i], 0.0, 1.0))
            signals[i] = position
            sizes[i] = weight if position else 1.0
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100., 102., 101., 99., 100., 99., 103., 100.]}, index=index)
    derived = pd.DataFrame({
        'atr': [1., 1., 4., 1., 1., 1., 1., 1.],
        'entry': [1, 0, 0, 0, -1, 0, 0, 0],
        'strength': [0.5] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, SignatureParams(stop_atr=2.0)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result.loc[result['signal'] != 0, 'size'], 0.6)
