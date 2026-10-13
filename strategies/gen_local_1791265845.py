from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpirationRepairParams:
    ma_bars: int = 20
    atr_bars: int = 14
    entry_z: float = 1.0
    trail_atr: float = 2.5
    hold_bars: int = 8
    size_floor: float = 0.20
    size_cap: float = 0.95
    size_full_z: float = 3.0


class GeneratedStrategy(BaseStrategy[ExpirationRepairParams]):
    strategy_id = 'gen_local_1791265845'

    @classmethod
    def params_type(cls) -> type[ExpirationRepairParams]:
        return ExpirationRepairParams

    def warmup_bars(self, params: ExpirationRepairParams) -> int:
        return max(max(2, int(params.ma_bars)) + 1,
                   max(1, int(params.atr_bars)) + 1)

    def indicators(self, data: pd.DataFrame, params: ExpirationRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        length = max(2, int(params.ma_bars))
        out['ma'] = close.rolling(length, min_periods=length).mean()
        out['std'] = close.rolling(length, min_periods=length).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['previous_z'] = out['z'].shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr_length = max(1, int(params.atr_bars))
        out['atr'] = out['true_range'].rolling(atr_length, min_periods=atr_length).mean()
        first_weekday = (data.index.dayofweek - (data.index.day - 1)) % 7
        out['expiration_day'] = 15 + (4 - first_weekday) % 7
        out['days_after_expiration'] = data.index.day - out['expiration_day']
        out['entry_window'] = out['days_after_expiration'].between(3, 5)
        threshold = max(0.01, float(params.entry_z))
        previous = out['previous_z']
        z = out['z']
        repairing = (z.abs() < previous.abs()) & (z * previous > 0)
        eligible = out['entry_window'] & repairing & (previous.abs() >= threshold) & (z.abs() >= threshold * 0.5)
        out['candidate'] = np.where(eligible & (z < 0), 1, np.where(eligible & (z > 0), -1, 0)).astype(int)
        floor = float(np.clip(params.size_floor, 0.01, 1.0))
        cap = float(np.clip(params.size_cap, floor, 1.0))
        full_z = max(threshold, float(params.size_full_z))
        out['conviction_size'] = (floor + (cap - floor) * (z.abs() / full_z).clip(0.0, 1.0)).fillna(floor)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ExpirationRepairParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        convictions = indicators['conviction_size'].fillna(0.20).to_numpy(dtype=float)
        position = 0
        entry_bar = 0
        watermark = 0.0
        stop = 0.0
        entry_size = 1.0
        multiplier = max(0.01, float(params.trail_atr))
        holding_limit = max(1, int(params.hold_bars))
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position = 0
                continue
            if position:
                # Longs track the highest close; shorts mirror this with the lowest close.
                signed_close = position * close
                watermark = max(watermark, signed_close)
                stop = max(stop, watermark - multiplier * atr)
                if signed_close <= stop or i - entry_bar >= holding_limit:
                    position = 0
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
                watermark = position * close
                stop = watermark - multiplier * atr
                entry_size = float(np.clip(convictions[i], 0.01, 1.0))
            signals[i] = position
            sizes[i] = entry_size if position else 1.0
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = ExpirationRepairParams()
    index = pd.date_range('2024-01-22', periods=6, freq='B')
    for direction in (1, -1):
        closes = np.array([100.0, 102.0, 105.0, 104.0, 102.0, 102.0])
        if direction == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({'close': closes}, index=index)
        derived = pd.DataFrame({
            'atr': 1.0,
            'candidate': [direction, 0, 0, 0, 0, 0],
            'conviction_size': 0.70,
        }, index=index)
        result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0]
        assert np.isclose(result['size'].iloc[1], 0.70)
        assert (result['size'] > 0).all()
