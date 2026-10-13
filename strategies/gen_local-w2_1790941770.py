from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MigrationParams:
    channel_bars: int = 30
    roc_bars: int = 5
    atr_bars: int = 14
    acceleration_threshold: float = 0.12
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    max_hold_bars: int = 10
    min_size: float = 0.20
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.channel_bars, self.roc_bars, self.atr_bars, self.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding limit must be positive.')
        if not all(np.isfinite(x) and x > 0 for x in (self.acceleration_threshold, self.breakeven_pct, self.trail_atr)):
            raise ValueError('Thresholds must be finite and positive.')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[MigrationParams]):
    strategy_id = 'gen_local-w2_1790941770'

    @classmethod
    def params_type(cls):
        return MigrationParams

    def warmup_bars(self, params: MigrationParams) -> int:
        return int(max(params.channel_bars + 2, params.roc_bars + 2, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: MigrationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['atr_fraction'] = out['atr'] / close.where(close > 0)
        out['channel_high'] = data['high'].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out['channel_low'] = data['low'].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        out['channel_width'] = out['channel_high'] - out['channel_low']
        out['position'] = (close - out['channel_low']) / out['channel_width'].where(out['channel_width'] > 0)
        out['migration'] = out['position'].diff()
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff() / out['atr_fraction'].where(out['atr_fraction'] > 0)
        out['directional_room'] = pd.Series(np.where(out['acceleration'] >= 0, 1 - out['position'], out['position']), index=data.index).clip(0, 1)
        out['entry_size'] = (out['acceleration'].abs() / params.acceleration_threshold * out['directional_room']).clip(params.min_size, params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MigrationParams) -> SignalFrame:
        required = ['atr', 'position', 'migration', 'roc', 'acceleration', 'entry_size']
        ready = indicators[required].notna().all(axis=1)
        ready &= indicators['atr'].gt(0)
        ready &= data[['open', 'close']].gt(0).all(axis=1)
        ready &= pd.Series(np.isfinite(data[['open', 'close']].to_numpy()).all(axis=1), index=data.index)
        long_entry = ready & indicators['position'].between(0.20, 0.50)
        long_entry &= indicators['migration'].gt(0) & indicators['roc'].gt(0)
        long_entry &= indicators['acceleration'].ge(params.acceleration_threshold)
        short_entry = ready & indicators['position'].between(0.50, 0.80)
        short_entry &= indicators['migration'].lt(0) & indicators['roc'].lt(0)
        short_entry &= indicators['acceleration'].le(-params.acceleration_threshold)
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        pending = False
        activated = False
        entry = stop = best = entry_atr = 0.0
        held_size = params.min_size
        age = 0
        for i in range(len(data)):
            if not bool(ready.iloc[i]):
                direction = 0
                pending = False
                continue
            close = float(data['close'].iloc[i])
            atr = float(indicators['atr'].iloc[i])
            if direction:
                if pending:
                    # The previous close's decision fills at this bar's open.
                    entry = float(data['open'].iloc[i])
                    best = direction * entry
                    stop = best - params.trail_atr * entry_atr
                    activated = False
                    age = 0
                    pending = False
                age += 1
                signed_close = direction * close
                # Close-based stop decisions execute on the following bar.
                exit_now = signed_close <= stop or age >= params.max_hold_bars
                if not exit_now:
                    best = max(best, signed_close)
                    if direction * (close - entry) / entry >= params.breakeven_pct:
                        activated = True
                    if activated:
                        # Signed coordinates make both long and short stops ratchet upward.
                        stop = max(stop, direction * entry, best - params.trail_atr * atr)
                        exit_now = signed_close <= stop
                if exit_now:
                    direction = 0
                    continue
            elif candidates[i]:
                direction = int(candidates[i])
                pending = True
                entry_atr = atr
                held_size = float(indicators['entry_size'].iloc[i])
            signals[i] = direction
            if direction:
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = MigrationParams()
    index = pd.date_range('2020-01-01', periods=4)
    for direction in (1, -1):
        closes = [100.0, 100.0 + direction * 3.0, 100.0, 100.0]
        data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': closes, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({
            'atr': 1.0,
            'position': 0.35 if direction == 1 else 0.65,
            'migration': direction * 0.05,
            'roc': direction * 0.01,
            'acceleration': [direction * 0.30, 0.0, 0.0, 0.0],
            'entry_size': 0.60,
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0]
        assert result['size'].gt(0).all()
        assert result['size'].iloc[1] == 0.60


if __name__ == '__main__':
    _self_check()
