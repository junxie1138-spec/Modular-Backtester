from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OverflowParams:
    baseline_bars: int = 20
    atr_bars: int = 14
    min_streak: int = 3
    efficiency_ratio: float = 0.55
    volume_ratio: float = 1.20
    range_ratio: float = 1.10
    trail_atr: float = 0.75
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OverflowParams]):
    strategy_id = 'gen_local_1791242303'

    @classmethod
    def params_type(cls) -> type[OverflowParams]:
        return OverflowParams

    def warmup_bars(self, params: OverflowParams) -> int:
        return int(max(params.baseline_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: OverflowParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        change = data['close'].diff()
        direction = np.sign(change).fillna(0).astype(int)
        groups = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(groups).cumcount().add(1).where(direction.ne(0), 0)
        candle_range = (data['high'] - data['low']).clip(lower=0)
        efficiency = (data['close'] - data['open']).abs().div(candle_range.replace(0, np.nan))
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            candle_range,
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        baseline_efficiency = efficiency.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean().shift(1)
        baseline_volume = data['volume'].rolling(params.baseline_bars, min_periods=params.baseline_bars).mean().shift(1)
        efficiency_fraction = efficiency.div(baseline_efficiency.replace(0, np.nan))
        volume_fraction = data['volume'].div(baseline_volume.where(baseline_volume.gt(0)))
        range_fraction = candle_range.div(atr.shift(1).where(atr.shift(1).gt(0)))
        ready = pd.concat([atr, efficiency_fraction, volume_fraction, range_fraction], axis=1)
        ready = ready.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & atr.gt(0)
        eligible = (
            ready
            & streak.ge(params.min_streak)
            & efficiency_fraction.le(params.efficiency_ratio)
            & volume_fraction.ge(params.volume_ratio)
            & range_fraction.ge(params.range_ratio)
        )
        onset = eligible & ~eligible.shift(1, fill_value=False)
        strength = (
            streak.div(max(params.min_streak, 1))
            * (1.0 - efficiency_fraction.clip(0.0, 1.0))
            * volume_fraction.clip(1.0, 2.0)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['direction'] = direction
        out['streak'] = streak.astype(int)
        out['range'] = candle_range.fillna(0.0)
        out['efficiency'] = efficiency.fillna(0.0)
        out['true_range'] = true_range.fillna(0.0)
        out['atr'] = atr.fillna(0.0)
        out['baseline_efficiency'] = baseline_efficiency.fillna(0.0)
        out['baseline_volume'] = baseline_volume.fillna(0.0)
        out['efficiency_fraction'] = efficiency_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['volume_fraction'] = volume_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['range_fraction'] = range_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['ready'] = ready
        out['eligible'] = eligible
        out['onset'] = onset
        out['entry'] = (-direction).where(onset, 0).astype(int)
        out['strength'] = strength
        out['entry_size'] = strength.div(2.0).clip(0.20, 1.0).astype(float)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OverflowParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        entry_sizes = indicators['entry_size'].to_numpy(dtype=float)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        trade_size = 1.0
        hold_limit = max(1, min(2, int(params.max_hold_bars)))
        for i in range(count):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0.0
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    # The first active bar follows the entry decision; its close
                    # initializes the in-trade extrema, excluding the signal bar.
                    high_water = close if age == 1 else max(high_water, close)
                    low_water = close if age == 1 else min(low_water, close)
                    if position == 1:
                        candidate = high_water - params.trail_atr * atr
                        stop = candidate if age == 1 else max(stop, candidate)
                        stopped = close <= stop
                    else:
                        # Mirror the highest-close long trail for short positions.
                        candidate = low_water + params.trail_atr * atr
                        stop = candidate if age == 1 else min(stop, candidate)
                        stopped = close >= stop
                    if stopped or age >= hold_limit:
                        position = 0
                # Exits emit zero for the next fill; no same-bar re-entry.
            elif valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                low_water = np.nan
                stop = np.nan
                trade_size = float(np.clip(entry_sizes[i], 0.20, 1.0))
            signals[i] = position
            sizes[i] = trade_size if position else 1.0
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = OverflowParams()
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({'open': [100.0] * 4, 'high': [102.0] * 4, 'low': [97.0] * 4, 'close': [100.0, 101.0, 99.0, 98.0], 'volume': [1000.0] * 4}, index=index)
    for side in (1, -1):
        derived = pd.DataFrame({'atr': [1.0] * 4, 'entry': [side, 0, 0, 0], 'entry_size': [0.4] * 4}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0]
        assert np.allclose(result['size'].to_numpy(), [1.0, 0.4, 0.4, 1.0])
        assert result['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived.isna().any().any()
    assert derived['entry'].eq(0).all()
    assert strategy.warmup_bars(params) >= max(params.baseline_bars, params.atr_bars) + 1
