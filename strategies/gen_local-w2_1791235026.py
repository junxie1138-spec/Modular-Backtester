from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    box_bars: int = 12
    baseline_bars: int = 60
    volume_bars: int = 20
    atr_bars: int = 14
    compression_ratio: float = 0.80
    max_box_atr: float = 5.5
    min_pressure: float = 0.12
    min_volume_ratio: float = 1.35
    breakeven_pct: float = 0.03
    initial_stop_atr: float = 2.0
    trail_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1791235026'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.box_bars, params.baseline_bars,
                       params.volume_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['daily_range'] = data['high'] - data['low']
        out['true_range'] = pd.concat([
            out['daily_range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_bars).mean()
        out['normalized_range'] = out['daily_range'] / data['close'].replace(0, np.nan)
        out['short_range'] = out['normalized_range'].rolling(p.box_bars).mean().shift(1)
        out['baseline_range'] = out['normalized_range'].rolling(p.baseline_bars).mean().shift(1)
        out['compression'] = out['short_range'] / out['baseline_range'].replace(0, np.nan)
        out['box_high'] = data['high'].rolling(p.box_bars).max().shift(1)
        out['box_low'] = data['low'].rolling(p.box_bars).min().shift(1)
        out['prior_atr'] = out['atr'].shift(1)
        out['box_width_atr'] = (out['box_high'] - out['box_low']) / out['prior_atr'].replace(0, np.nan)
        out['volume_baseline'] = data['volume'].rolling(p.volume_bars).mean().shift(1)
        out['volume_ratio'] = data['volume'] / out['volume_baseline'].replace(0, np.nan)
        out['close_location'] = ((2 * data['close'] - data['high'] - data['low'])
                                 / out['daily_range'].replace(0, np.nan)).fillna(0).clip(-1, 1)
        out['signed_volume'] = out['close_location'] * data['volume']
        out['pressure_numerator'] = out['signed_volume'].rolling(p.box_bars).sum().shift(1)
        out['pressure_denominator'] = data['volume'].rolling(p.box_bars).sum().shift(1)
        out['pressure'] = out['pressure_numerator'] / out['pressure_denominator'].replace(0, np.nan)
        required = ['atr', 'compression', 'box_width_atr', 'volume_ratio', 'pressure', 'box_high']
        out['ready'] = (np.isfinite(out[required]).all(axis=1)
                        & np.isfinite(data).all(axis=1)
                        & (data['open'] > 0) & (data['close'] > 0)
                        & (data['volume'] > 0) & (out['atr'] > 0))
        out['entry'] = (out['ready']
                        & (out['compression'] <= p.compression_ratio)
                        & (out['box_width_atr'] <= p.max_box_atr)
                        & (out['pressure'] >= p.min_pressure)
                        & (out['volume_ratio'] >= p.min_volume_ratio)
                        & (data['close'] > out['box_high'])
                        & (data['close'] > data['open']))
        out['strength'] = (0.5 * ((out['volume_ratio'] - 1) / 2).clip(0, 1)
                           + 0.5 * (out['pressure'] / 0.6).clip(0, 1))
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        p = params
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), p.min_size, dtype=float)
        active = False
        pending = False
        armed = False
        entry_price = stop = peak = 0.0
        trade_size = p.min_size
        age = 0
        for i in range(len(data)):
            if pending:
                # The previous close's decision fills at this bar's open.
                entry_price = float(data['open'].iloc[i])
                entry_atr = float(indicators['atr'].iloc[i - 1])
                pending = False
                if not np.isfinite(entry_price) or entry_price <= 0:
                    continue
                active = True
                armed = False
                age = 0
                peak = entry_price
                stop = entry_price - p.initial_stop_atr * entry_atr
            if active:
                age += 1
                close = float(data['close'].iloc[i])
                atr = float(indicators['atr'].iloc[i])
                # Stops are evaluated at the close; liquidation fills next bar.
                if not np.isfinite(close) or close <= stop or age >= p.hold_bars:
                    active = False
                    continue
                peak = max(peak, close)
                if close >= entry_price * (1 + p.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, peak - p.trail_atr * atr)
                if close <= stop:
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = trade_size
            elif bool(indicators['ready'].iloc[i]) and bool(indicators['entry'].iloc[i]):
                strength = float(indicators['strength'].iloc[i])
                trade_size = p.min_size + (p.max_size - p.min_size) * float(np.clip(strength, 0, 1))
                signals[i] = 1
                sizes[i] = trade_size
                pending = True
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    sample = pd.DataFrame({
        'open': [100., 100., 100., 104., 105., 104., 102.],
        'high': [101., 101., 105., 106., 106., 105., 103.],
        'low': [99., 99., 99., 103., 103., 101., 100.],
        'close': [100., 100., 104., 105., 104., 102., 101.],
        'volume': [1000.] * 7,
    }, index=index)
    features = pd.DataFrame({
        'ready': [True] * 7,
        'entry': [True, False, False, False, False, False, False],
        'atr': [1., 1., 1., 1., 10., 1., 1.],
        'strength': [0.8] * 7,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, features, None, CompressionParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0]
    assert np.allclose(result.loc[result['signal'] == 1, 'size'], 0.81)
    assert (result['size'] > 0).all()
