from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.4
    fractional_window: int = 32
    scale_window: int = 20
    compression_window: int = 20
    compression_ratio: float = 0.7
    acceleration_trigger: float = 1.0
    atr_window: int = 14
    breakeven_pct: float = 0.005
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791252015'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fractional_window + params.scale_window + 2,
                       params.compression_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError('fractional_order must be between zero and one')
        if min(params.fractional_window, params.scale_window,
               params.compression_window, params.atr_window) < 3:
            raise ValueError('lookback windows must be at least three')
        if min(params.compression_ratio, params.acceleration_trigger,
               params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError('thresholds and stop parameters must be positive')
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, len(weights)):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        # ponytail: fixed truncation retains residual level memory; increase the window for longer memory.
        out['fractional_close'] = out['log_close'].copy()
        for lag in range(1, len(weights)):
            out['fractional_close'] += weights[lag] * out['log_close'].shift(lag)
        out['fractional_impulse'] = out['fractional_close'].diff()
        out['fractional_acceleration'] = out['fractional_impulse'].diff()
        out['impulse_scale'] = out['fractional_impulse'].rolling(
            params.scale_window, min_periods=params.scale_window
        ).std(ddof=0).shift(1)
        out['acceleration_score'] = out['fractional_acceleration'].div(
            out['impulse_scale'].where(out['impulse_scale'] > 0)
        )
        out['relative_range'] = (data['high'] - data['low']).div(
            data['close'].where(data['close'] > 0)
        )
        out['short_range'] = out['relative_range'].rolling(3, min_periods=3).mean()
        out['baseline_range'] = out['relative_range'].rolling(
            params.compression_window, min_periods=params.compression_window
        ).mean()
        out['compression'] = out['short_range'].div(
            out['baseline_range'].where(out['baseline_range'] > 0)
        ).shift(1)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['conviction'] = (0.25 + 0.35 * out['acceleration_score'].div(
            params.acceleration_trigger
        ).clip(lower=0, upper=2)).clip(0.25, 0.95).fillna(0.25)
        out['entry'] = (
            out['compression'].le(params.compression_ratio)
            & out['fractional_impulse'].gt(0)
            & out['acceleration_score'].gt(params.acceleration_trigger)
            & out['acceleration_score'].shift(1).le(params.acceleration_trigger)
            & out['atr'].gt(0)
        ).fillna(False)
        out.loc[out.index[:self.warmup_bars(params)], 'entry'] = False
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({'signal': 0, 'size': 0.25}, index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].fillna(0.25).to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        pending = False
        active = False
        armed = False
        entry_price = stop = watermark = allocation = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    watermark = entry_price
                    stop = entry_price - params.trail_atr * atr[i - 1]
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                valid = np.isfinite(closes[i]) and np.isfinite(highs[i])
                if valid:
                    watermark = max(watermark, highs[i])
                    armed = armed or watermark >= entry_price * (1 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, watermark - params.trail_atr * atr[i])
                # Stops are evaluated at the close; the shifted exit fills at the next open.
                if not valid or closes[i] <= stop or age >= 2:
                    active = False
                else:
                    decisions[i] = 1
                    sizes[i] = allocation
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                allocation = float(np.clip(conviction[i], 0.25, 0.95))
                decisions[i] = 1
                sizes[i] = allocation
                pending = True
        df['signal'] = decisions
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=5)
    bars = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [100.0, 100.0, 102.0, 101.0, 100.0],
        'low': [99.0, 99.0, 100.0, 99.0, 99.0],
        'close': [100.0, 100.0, 101.0, 99.5, 100.0],
        'volume': [1000.0] * 5
    }, index=index)
    derived = pd.DataFrame({
        'entry': [False, True, False, False, False],
        'conviction': [0.8] * 5,
        'atr': [1.0] * 5
    }, index=index)
    result = GeneratedStrategy().generate_signals(bars, derived, None, Params()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert result['size'].iloc[2] == 0.8 and result['size'].gt(0).all()
