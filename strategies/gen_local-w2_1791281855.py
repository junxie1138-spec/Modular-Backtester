from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    short_window: int = 5
    baseline_window: int = 30
    atr_window: int = 14
    compression_ratio: float = 0.70
    downside_share: float = 0.20
    upside_retention: float = 0.35
    stop_atr: float = 1.50
    hold_days: int = 4
    min_size: float = 0.25
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791281855'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.short_window, params.baseline_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.short_window, params.baseline_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars.')
        if not (0 < params.compression_ratio < 1 and 0 < params.downside_share < 1):
            raise ValueError('Compression and downside thresholds must lie between zero and one.')
        if not (0 < params.upside_retention <= 1):
            raise ValueError('Upside retention must lie between zero and one.')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_energy'] = out['return'].pow(2)
        out['down_energy'] = out['return'].clip(upper=0).pow(2)
        out['up_energy'] = out['return'].clip(lower=0).pow(2)
        for name in ('return_energy', 'down_energy', 'up_energy'):
            out['short_' + name] = out[name].rolling(params.short_window, min_periods=params.short_window).mean().shift(1)
            out['baseline_' + name] = out[name].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(1)
        out['compression'] = out['short_return_energy'] / out['baseline_return_energy'].replace(0, np.nan)
        out['down_share'] = out['short_down_energy'] / out['short_return_energy'].replace(0, np.nan)
        out['up_retention'] = out['short_up_energy'] / out['baseline_up_energy'].replace(0, np.nan)
        out['baseline_rms'] = np.sqrt(out['baseline_return_energy'])
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['strength'] = (1 - out['down_share'] / params.downside_share).clip(0, 1)
        out['entry'] = (
            (out['compression'] < params.compression_ratio)
            & (out['down_share'] < params.downside_share)
            & (out['up_retention'] >= params.upside_retention)
            & (out['return'] > 0)
            & (out['return'] <= out['baseline_rms'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        if not (3 <= params.hold_days <= 5 and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError('Use a three-to-five-session hold and a positive finite ATR stop.')
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError('Position sizes must satisfy 0 < min_size <= max_size <= 1.')
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].fillna(0).clip(0, 1).to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        position_size = params.min_size
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Freeze the stop at the next-open entry using decision-bar ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                invalid = not (np.isfinite(closes[i]) and np.isfinite(stop))
                expired = i - entry_bar + 1 >= params.hold_days
                if invalid or closes[i] < stop or expired:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                position_size = params.min_size + (params.max_size - params.min_size) * strength[i]
                signals[i] = 1
                sizes[i] = position_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0, 102.0, 100.0, 100.0, 100.0, 100.0],
        'high': [103.0] * 6,
        'low': [97.0] * 6,
        'close': [100.0, 101.0, 98.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2.0] * 6,
        'entry': [True, False, False, False, False, False],
        'strength': [0.8] * 6,
    }, index=index)
    params = CompressionParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert np.isclose(result['size'].iloc[1], 0.85)
    assert (result['size'] > 0).all()
    data['open'] = 100.0
    data['close'] = 100.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
