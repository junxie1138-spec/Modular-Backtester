from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    topology_threshold: float = 0.30
    pressure_window: int = 5
    pressure_threshold: float = 0.20
    reversal_threshold: float = 0.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791314372'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window, params.pressure_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.pressure_window < 1:
            raise ValueError('Indicator windows must be at least 3 and 1 respectively.')
        if not 0.0 < params.topology_threshold <= 1.0:
            raise ValueError('topology_threshold must be in (0, 1].')
        if not 0.0 < params.pressure_threshold <= 1.0:
            raise ValueError('pressure_threshold must be in (0, 1].')
        if not 0.0 < params.reversal_threshold <= 1.0:
            raise ValueError('reversal_threshold must be in (0, 1].')

        result = pd.DataFrame(index=data.index)
        price = np.log(data['close'].where(data['close'] > 0)).replace([np.inf, -np.inf], np.nan)
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        maximum_slope = pd.Series(-np.inf, index=data.index)
        minimum_slope = pd.Series(np.inf, index=data.index)

        # An edge exists when its backward slope clears every intervening slope.
        for lag in range(1, params.graph_window):
            slope = (price.shift(lag) - price) / float(lag)
            upper_degree += (slope > maximum_slope).astype(float)
            lower_degree += (slope < minimum_slope).astype(float)
            maximum_slope = np.maximum(maximum_slope, slope)
            minimum_slope = np.minimum(minimum_slope, slope)

        graph_ready = price.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        result['upper_degree'] = upper_degree.where(graph_ready)
        result['lower_degree'] = lower_degree.where(graph_ready)
        result['topology_bias'] = (result['upper_degree'] - result['lower_degree']) / float(params.graph_window - 2)

        spread = (data['high'] - data['low']).where(data['high'] > data['low'])
        location = ((2.0 * data['close'] - data['high'] - data['low']) / spread).clip(-1.0, 1.0)
        location = location.mask(data['high'].eq(data['low']), 0.0).replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].where(data['volume'] >= 0).replace([np.inf, -np.inf], np.nan)
        result['close_location'] = location
        result['signed_volume'] = location * volume
        result['prior_signed_volume'] = result['signed_volume'].shift(1).rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        result['prior_volume'] = volume.shift(1).rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        result['prior_pressure'] = result['prior_signed_volume'] / result['prior_volume'].where(result['prior_volume'] > 0)

        long_entry = (
            result['topology_bias'].ge(params.topology_threshold)
            & result['prior_pressure'].le(-params.pressure_threshold)
            & result['close_location'].ge(params.reversal_threshold)
        )
        short_entry = (
            result['topology_bias'].le(-params.topology_threshold)
            & result['prior_pressure'].ge(params.pressure_threshold)
            & result['close_location'].le(-params.reversal_threshold)
        )
        result['entry_direction'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('Holding period and profit target must be positive.')
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        fill_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position != 0:
                # The preceding close queued this entry for the current open.
                if i == fill_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    held_bars = i - fill_bar + 1
                    if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.max_hold_bars:
                        position = 0
                decisions[i] = position
                continue

            if entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                fill_bar = i + 1
                entry_price = np.nan
            decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100.0] * 7,
        'high': [104.0] * 7,
        'low': [99.0] * 7,
        'close': [100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({'entry_direction': [1, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = VisibilityParams(max_hold_bars=2)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
