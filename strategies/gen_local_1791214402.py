from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    shape_bars: int = 5
    history_bars: int = 252
    neighbor_percentile: float = 0.10
    entry_percentile: float = 0.55
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if not (
            self.shape_bars >= 2 and self.history_bars >= 10
            and self.atr_bars >= 1 and self.hold_bars >= 1
            and 0.0 < self.neighbor_percentile < 1.0
            and 0.0 < self.entry_percentile < 1.0
            and np.isfinite(self.stop_atr) and self.stop_atr > 0.0
        ):
            raise ValueError('Invalid range-analog parameters')


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = 'gen_local_1791214402'

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    def warmup_bars(self, params):
        max_lag = params.hold_bars + params.shape_bars + params.history_bars - 1
        return int(max(params.shape_bars + max_lag, params.atr_bars + 1))

    def indicators(self, data, params):
        result = pd.DataFrame(index=data.index)
        result['range_fraction'] = (data['high'] - data['low']).clip(lower=0).div(
            data['close'].replace(0.0, np.nan)
        )
        shape = pd.concat(
            [result['range_fraction'].shift(j) for j in range(params.shape_bars)],
            axis=1,
        )
        result['shape_mean'] = shape.mean(axis=1, skipna=False)
        shape = shape.div(result['shape_mean'].replace(0.0, np.nan), axis=0)
        shape.columns = [f'range_shape_{j}' for j in range(params.shape_bars)]
        result = pd.concat([result, shape], axis=1)
        previous_close = data['close'].shift(1)
        result['true_range'] = pd.concat(
            [
                data['high'] - data['low'],
                (data['high'] - previous_close).abs(),
                (data['low'] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        result['atr'] = result['true_range'].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        values = shape.to_numpy(dtype=float)
        n = len(data)
        distances = np.full((n, params.history_bars), np.nan)
        outcomes = np.full_like(distances, np.nan)
        # Historical outcomes finish before the current shape starts.
        first_lag = params.hold_bars + params.shape_bars
        for column, lag in enumerate(range(first_lag, first_lag + params.history_bars)):
            if lag < n:
                distances[lag:, column] = np.square(
                    values[lag:] - values[:-lag]
                ).mean(axis=1)
            outcomes[:, column] = (
                data['close'].shift(lag - params.hold_bars).div(
                    data['open'].shift(lag - 1).replace(0.0, np.nan)
                ) - 1.0
            ).to_numpy(dtype=float)
        ready = np.isfinite(distances).all(axis=1) & np.isfinite(outcomes).all(axis=1)
        distance_cutoff = np.quantile(distances, params.neighbor_percentile, axis=1)
        selected = (distances <= distance_cutoff[:, None]) & np.isfinite(outcomes)
        count = selected.sum(axis=1)
        forecast = np.divide(
            np.where(selected, outcomes, 0.0).sum(axis=1),
            count,
            out=np.full(n, np.nan),
            where=count > 0,
        )
        result['distance_cutoff'] = distance_cutoff
        result['analog_count'] = count
        result['analog_return'] = np.where(ready, forecast, np.nan)
        result['entry_threshold'] = np.where(
            ready, np.quantile(outcomes, params.entry_percentile, axis=1), np.nan
        )
        result['entry'] = (
            (result['analog_return'] > result['entry_threshold'])
            & (result['analog_return'] > 0.0)
            & (result['atr'] > 0.0)
        ).fillna(False)
        return result

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame({'signal': 0, 'size': 1.0}, index=data.index)
        target = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        frozen_atr = np.nan
        stop_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Freeze the decision-bar ATR around the next bar's actual open.
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                if (
                    not np.isfinite(stop_price)
                    or not np.isfinite(closes[i])
                    or closes[i] < stop_price
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
                target[i] = int(active)
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                target[i] = 1
        df['signal'] = target
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeAnalogParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame(
        {'open': 10.0, 'high': 11.0, 'low': 7.0, 'close': 10.0, 'volume': 100.0},
        index=index,
    )
    data.loc[index[2], 'close'] = 7.0
    indicators = pd.DataFrame({'atr': 1.0, 'entry': False}, index=index)
    indicators.loc[index[0], 'entry'] = True
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].tolist() == [0, 1, 1] + [0] * 9
    data['close'] = 10.0
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].tolist() == [0] + [1] * params.hold_bars + [0] * 3


if __name__ == '__main__':
    _self_check()
