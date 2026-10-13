from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCapacityParams:
    lookback: int = 126
    range_threshold: float = 1.6
    pressure_threshold: float = 0.5
    profit_target: float = 0.012
    hold_bars: int = 4
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeCapacityParams]):
    strategy_id = 'gen_local_1791219033'

    @classmethod
    def params_type(cls) -> type[RangeCapacityParams]:
        return RangeCapacityParams

    def warmup_bars(self, params: RangeCapacityParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: RangeCapacityParams) -> pd.DataFrame:
        window = int(params.lookback)
        spread = data['high'] - data['low']
        valid = (data['close'] > 0) & (spread > 0)
        normalized_range = (spread / data['close'].where(data['close'] > 0)).where(valid)
        market_expected = normalized_range.shift(1).rolling(window, min_periods=window).mean()
        weekday = data.index.dayofweek
        expected = pd.Series(np.nan, index=data.index, dtype=float)
        for day in range(7):
            mask = weekday == day
            seasonal = normalized_range.where(mask).shift(1).rolling(window, min_periods=8).mean()
            expected.loc[mask] = seasonal.loc[mask]
        range_ratio = normalized_range / expected.where(expected > 0)
        capacity = expected / market_expected.where(market_expected > 0)
        pressure = ((2.0 * data['close'] - data['high'] - data['low']) / spread.where(spread > 0)).clip(-1.0, 1.0)
        ready = valid & expected.notna() & market_expected.notna() & range_ratio.notna() & pressure.notna()
        result = pd.DataFrame({
            'normalized_range': normalized_range,
            'weekday_expected_range': expected,
            'market_expected_range': market_expected,
            'weekday_capacity': capacity,
            'range_ratio': range_ratio,
            'closing_pressure': pressure,
            'ready': ready.astype(float),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeCapacityParams) -> SignalFrame:
        if params.lookback < 40 or params.range_threshold <= 1.0:
            raise ValueError('lookback must be >= 40 and range_threshold must exceed 1')
        if not 0.0 < params.pressure_threshold <= 1.0:
            raise ValueError('pressure_threshold must be in (0, 1]')
        if params.hold_bars not in (3, 4, 5) or params.profit_target <= 0.0:
            raise ValueError('hold_bars must be 3, 4, or 5 and profit_target must be positive')
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError('sizes must satisfy 0 < min_size <= max_size <= 1')

        ratio = indicators['range_ratio']
        pressure = indicators['closing_pressure']
        eligible = (
            indicators['ready'].eq(1.0)
            & indicators['weekday_capacity'].gt(0.0)
            & indicators['weekday_capacity'].le(1.0)
            & ratio.ge(params.range_threshold)
            & pressure.abs().ge(params.pressure_threshold)
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & pressure.lt(0.0)] = 1
        candidates.loc[eligible & pressure.gt(0.0)] = -1
        strength = ((ratio / params.range_threshold - 1.0) * pressure.abs()).clip(0.0, 1.0)
        candidate_size = params.min_size + (params.max_size - params.min_size) * strength

        raw_signal = np.zeros(len(data), dtype=int)
        raw_size = np.full(len(data), params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = candidates.to_numpy(dtype=int)
        sizes = candidate_size.to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        position_size = params.min_size

        for i in range(len(data)):
            if position:
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                expired = i - entry_bar + 1 >= params.hold_bars
                if (np.isfinite(gain) and gain >= params.profit_target) or expired:
                    position = 0
                    entry_price = np.nan
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                position_size = float(sizes[i])
                entry_price = np.nan
            raw_signal[i] = position
            if position:
                raw_size[i] = position_size

        df = pd.DataFrame({'signal': raw_signal, 'size': raw_size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'ready': 1.0, 'weekday_capacity': 0.8, 'range_ratio': 2.0, 'closing_pressure': -0.8}, index=index)
    params = RangeCapacityParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].iloc[:6].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].between(params.min_size, params.max_size).all()
    data.loc[index[2], 'close'] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].iloc[:4].tolist() == [0, 1, 1, 0]
    indicators['closing_pressure'] = 0.8
    data.loc[index[2], 'close'] = 98.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result['signal'].iloc[:4].tolist() == [0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
