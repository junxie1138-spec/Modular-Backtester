from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalCompressionParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    compression_window: int = 10
    percentile_window: int = 126
    compression_percentile: float = 0.25
    tail_percentile: float = 0.20
    atr_window: int = 14
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[FractionalCompressionParams]):
    strategy_id = 'gen_local_1791344172'

    @classmethod
    def params_type(cls):
        return FractionalCompressionParams

    def warmup_bars(self, params: FractionalCompressionParams) -> int:
        return int(max(
            params.fractional_window + params.percentile_window + 1,
            params.compression_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalCompressionParams) -> pd.DataFrame:
        if min(params.fractional_window, params.compression_window,
               params.percentile_window, params.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must lie between zero and one')
        if not 0.0 < params.compression_percentile < 1.0:
            raise ValueError('Invalid compression percentile')
        if not 0.0 < params.tail_percentile < 0.5:
            raise ValueError('tail_percentile must lie between zero and 0.5')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or params.hold_bars < 1:
            raise ValueError('Trailing distance and holding period must be positive')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            high - low,
            (high - out['previous_close']).abs(),
            (low - out['previous_close']).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['range_high'] = high.rolling(params.compression_window).max()
        out['range_low'] = low.rolling(params.compression_window).min()
        out['range_width'] = (
            (out['range_high'] - out['range_low']) / close.where(close > 0)
        )
        out['compression_threshold'] = out['range_width'].rolling(
            params.percentile_window
        ).quantile(params.compression_percentile).shift(1)

        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = weights[lag - 1] * (
                lag - 1 - params.fractional_order
            ) / lag
        # ponytail: finite fractional memory; increase fractional_window if tail truncation matters.
        fractional_close = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional_close = fractional_close + close.shift(lag) * weights[lag]
        out['fractional_close'] = fractional_close
        out['fractional_upper'] = fractional_close.rolling(
            params.percentile_window
        ).quantile(1.0 - params.tail_percentile).shift(1)
        out['fractional_lower'] = fractional_close.rolling(
            params.percentile_window
        ).quantile(params.tail_percentile).shift(1)
        out['compressed'] = (
            out['range_width'] <= out['compression_threshold']
        ).fillna(False)
        out['expanding'] = (
            out['range_width'] > out['range_width'].shift(1)
        ).fillna(False)
        out['upper_cross'] = (
            (fractional_close > out['fractional_upper'])
            & (fractional_close.shift(1) <= out['fractional_upper'].shift(1))
        ).fillna(False)
        out['lower_cross'] = (
            (fractional_close < out['fractional_lower'])
            & (fractional_close.shift(1) >= out['fractional_lower'].shift(1))
        ).fillna(False)
        release = out['compressed'].shift(1, fill_value=False) & out['expanding']
        out['long_entry'] = release & out['upper_cross']
        out['short_entry'] = release & out['lower_cross']
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalCompressionParams,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        long_entry = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = low_water = high_stop = short_stop = 0.0

        for i in range(len(data)):
            price = close[i]
            if not np.isfinite(price) or price <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                continue
            distance = params.trail_atr * atr[i]
            if position:
                high_water = max(high_water, price)
                high_stop = max(high_stop, high_water - distance)
                low_water = min(low_water, price)
                short_stop = min(short_stop, low_water + distance)
                # Required high-water drawdown exit applies to both directions.
                # Shorts also have a downward-ratcheting stop against rebounds.
                if (price <= high_stop
                        or (position == -1 and price >= short_stop)
                        or i - entry_bar >= params.hold_bars):
                    position = 0
                    continue
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                entry_bar = i
                high_water = low_water = price
                high_stop = price - distance
                short_stop = price + distance
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = FractionalCompressionParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100., 103., 100., 100., 100., 99., 97., 97.]}, index=index)
    indicators = pd.DataFrame({
        'atr': 1.0,
        'long_entry': [True, False, False, False, False, False, False, False],
        'short_entry': [False, False, False, False, True, False, False, False],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert (result['size'] > 0).all()
    params.hold_bars = 2
    data['close'] = 100.0
    indicators['short_entry'] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
