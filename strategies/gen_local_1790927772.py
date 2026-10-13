from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    level_span: int = 8
    slope_span: int = 5
    atr_window: int = 14
    pressure_window: int = 3
    innovation_threshold: float = 0.04
    pressure_threshold: float = 0.20
    trail_atr: float = 0.80
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790927772'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.level_span, params.slope_span, params.atr_window, params.pressure_window) < 1:
            raise ValueError('Indicator lengths must be positive.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        if not all(np.isfinite(v) and v > 0 for v in (params.innovation_threshold, params.pressure_threshold, params.trail_atr)):
            raise ValueError('Thresholds and trail_atr must be finite and positive.')
        return int(max(params.level_span + params.slope_span + 1, params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        high = data['high'].replace([np.inf, -np.inf], np.nan)
        low = data['low'].replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].replace([np.inf, -np.inf], np.nan).clip(lower=0)
        previous = close.shift(1)
        out['true_range'] = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean().where(lambda x: x > 0)
        out['level'] = close.ewm(span=params.level_span, adjust=False, min_periods=params.level_span).mean()
        out['level_change'] = out['level'].diff()
        out['slope'] = out['level_change'].ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        out['innovation'] = (out['level_change'] - out['slope'].shift(1)) / out['atr']
        daily_range = (high - low).where(high > low)
        out['closing_location'] = ((2 * close - high - low) / daily_range).clip(-1, 1)
        out.loc[(high == low) & close.notna(), 'closing_location'] = 0.0
        out['signed_volume'] = out['closing_location'] * volume
        out['pressure_volume'] = volume.rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['pressure'] = out['signed_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum() / out['pressure_volume'].where(out['pressure_volume'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        self.warmup_bars(params)
        valid = indicators[['atr', 'slope', 'innovation', 'pressure']].notna().all(axis=1) & data['close'].notna()
        long_entry = valid & (indicators['slope'] > 0) & (indicators['innovation'] > params.innovation_threshold) & (indicators['pressure'] > params.pressure_threshold)
        short_entry = valid & (indicators['slope'] < 0) & (indicators['innovation'] < -params.innovation_threshold) & (indicators['pressure'] < -params.pressure_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                age += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    position = 0
                else:
                    # The first held close follows the entry decision by one bar.
                    if position == 1:
                        high_water = close if np.isnan(high_water) else max(high_water, close)
                        candidate = high_water - params.trail_atr * atr
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        stopped = close <= stop
                    else:
                        low_water = close if np.isnan(low_water) else min(low_water, close)
                        candidate = low_water + params.trail_atr * atr
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        stopped = close >= stop
                    if stopped or age >= params.hold_bars:
                        position = 0
                # An exit remains flat for this decision; no immediate re-entry.
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water = np.nan
                low_water = np.nan
                stop = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [100., 102., 100., 100., 100.], 'high': [101., 103., 101., 101., 101.], 'low': [99., 101., 99., 99., 99.], 'close': [100., 102., 100., 100., 100.], 'volume': 1000.}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'slope': 1.0, 'innovation': [1., 0., 0., 0., 0.], 'pressure': 1.0}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    derived['pressure'] = -1.0
    assert strategy.generate_signals(data, derived, None, params).data['signal'].eq(0).all()
    derived['slope'] = -1.0
    derived['innovation'] = [-1., 0., 0., 0., 0.]
    assert strategy.generate_signals(data, derived, None, params).data['signal'].tolist() == [0, -1, -1, 0, 0]
    assert strategy.indicators(data, params).index.equals(index)
