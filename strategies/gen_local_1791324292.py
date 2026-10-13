from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CongestionParams:
    compression_bars: int = 3
    volume_lookback: int = 20
    atr_lookback: int = 14
    compression_ratio: float = 1.4
    volume_multiple: float = 1.2
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[CongestionParams]):
    strategy_id = 'gen_local_1791324292'

    @classmethod
    def params_type(cls) -> type[CongestionParams]:
        return CongestionParams

    @staticmethod
    def warmup_bars(params: CongestionParams) -> int:
        return int(max(params.volume_lookback, params.atr_lookback) + params.compression_bars + 3)

    def indicators(self, data: pd.DataFrame, params: CongestionParams) -> pd.DataFrame:
        if min(params.compression_bars, params.volume_lookback, params.atr_lookback) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        values = [params.compression_ratio, params.volume_multiple, params.breakeven_pct, params.trail_atr]
        if not np.isfinite(values).all() or min(values) <= 0 or not 3 <= params.holding_bars <= 5:
            raise ValueError('Positive finite thresholds and a three-to-five-bar holding limit are required')
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        volume = data['volume'].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError('OHLC prices must be positive and finite; volume must be finite and nonnegative')
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_lookback, min_periods=params.atr_lookback).mean()
        out['background_atr'] = out['atr'].shift(params.compression_bars).replace(0, np.nan)
        out['background_volume'] = data['volume'].rolling(params.volume_lookback, min_periods=params.volume_lookback).mean().shift(params.compression_bars).replace(0, np.nan)
        out['box_high'] = data['high'].rolling(params.compression_bars, min_periods=params.compression_bars).max()
        out['box_low'] = data['low'].rolling(params.compression_bars, min_periods=params.compression_bars).min()
        out['box_width'] = out['box_high'] - out['box_low']
        out['box_volume'] = data['volume'].rolling(params.compression_bars, min_periods=params.compression_bars).mean()
        out['relative_volume'] = data['volume'] / out['background_volume']
        out['move'] = data['close'].diff()
        out['impact'] = out['move'].abs() / out['atr'].replace(0, np.nan) / out['relative_volume'].replace(0, np.nan)
        out['congestion'] = ((out['box_width'] <= params.compression_ratio * out['background_atr']) & (out['box_volume'] >= out['background_volume'])).fillna(False)
        out['volume_confirmed'] = (out['relative_volume'] >= params.volume_multiple).fillna(False)
        out['first_long'] = (out['congestion'].shift(1, fill_value=False) & out['volume_confirmed'] & (data['close'] > out['box_high'].shift(1)) & (out['move'] > 0) & (data['close'] > data['open'])).fillna(False)
        out['first_short'] = (out['congestion'].shift(1, fill_value=False) & out['volume_confirmed'] & (data['close'] < out['box_low'].shift(1)) & (out['move'] < 0) & (data['close'] < data['open'])).fillna(False)
        accelerating = out['impact'] > out['impact'].shift(1)
        out['long_entry'] = (out['first_long'].shift(1, fill_value=False) & out['volume_confirmed'] & accelerating & (out['move'] > 0) & (data['close'] > data['open']) & (data['close'] > out['box_high'].shift(2))).fillna(False)
        out['short_entry'] = (out['first_short'].shift(1, fill_value=False) & out['volume_confirmed'] & accelerating & (out['move'] < 0) & (data['close'] < data['open']) & (data['close'] < out['box_low'].shift(2))).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CongestionParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].to_numpy(dtype=bool)
        shorts = indicators['short_entry'].to_numpy(dtype=bool)
        side = pending = age = 0
        entry = stop = 0.0
        armed = False
        for i in range(len(data)):
            if pending:
                side = pending
                pending = 0
                entry = opens[i]
                # Signed price rises with profit for both directions.
                stop = side * entry - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
            if side:
                age += 1
                signed_close = side * closes[i]
                # Stops are close-based decisions, executed next bar.
                # Test the existing stop before tightening it at this close.
                if signed_close <= stop or age >= params.holding_bars or not np.isfinite(atr[i]):
                    side = 0
                    continue
                if side * (closes[i] - entry) / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry, signed_close - params.trail_atr * atr[i])
                signals[i] = side
            elif i >= self.warmup_bars(params) - 1 and np.isfinite(atr[i]) and atr[i] > 0:
                pending = 1 if longs[i] else (-1 if shorts[i] else 0)
                signals[i] = pending
        df = pd.DataFrame({'signal': signals, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = CongestionParams(volume_lookback=10, atr_lookback=10)
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=32, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 100.0, 'volume': 100.0}, index=index)
    data.iloc[20:23] = [100.0, 100.2, 99.8, 100.0, 150.0]
    data.iloc[23] = [100.0, 101.2, 99.8, 101.0, 150.0]
    data.iloc[24] = [101.0, 103.2, 100.8, 103.0, 150.0]
    data.iloc[25] = [103.0, 105.2, 102.8, 105.0, 150.0]
    data.iloc[26] = [105.0, 105.2, 101.8, 102.0, 100.0]
    for direction in (1, -1):
        sample = data.copy()
        if direction == -1:
            sample['open'] = 200.0 - data['open']
            sample['close'] = 200.0 - data['close']
            sample['high'] = 200.0 - data['low']
            sample['low'] = 200.0 - data['high']
        derived = strategy.indicators(sample, params)
        result = strategy.generate_signals(sample, derived, None, params).data
        assert result['signal'].iloc[:25].eq(0).all()
        assert result['signal'].iloc[25] == direction
        assert result['signal'].iloc[26] == direction
        assert result['signal'].iloc[27] == 0
        assert result['size'].gt(0).all()
        prefix = sample.iloc[:26]
        prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
        assert result.iloc[:26].equals(prefix_result)


if __name__ == '__main__':
    _self_check()
