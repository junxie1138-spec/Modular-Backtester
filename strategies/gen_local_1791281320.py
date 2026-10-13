from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    shape_bars: int = 6
    history_bars: int = 252
    neighbors: int = 20
    horizon_days: int = 4
    entry_edge: float = 0.002

    def __post_init__(self):
        if self.shape_bars < 2 or self.neighbors < 1:
            raise ValueError('Shape length must be at least two and neighbors positive.')
        if self.horizon_days < 3 or self.horizon_days > 5:
            raise ValueError('Forecast horizon must be three to five sessions.')
        if self.history_bars < self.neighbors + self.horizon_days + self.shape_bars:
            raise ValueError('History must accommodate enough matured analogs.')
        if not np.isfinite(self.entry_edge) or self.entry_edge < 0:
            raise ValueError('Entry edge must be finite and nonnegative.')


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791281320'

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.history_bars + params.shape_bars + params.horizon_days + 2)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_price'] = np.log(data['close'].where(data['close'] > 0))
        valid_range = (data['low'] > 0) & (data['high'] >= data['low'])
        out['log_range'] = np.log((data['high'] / data['low']).where(valid_range))
        out['price_step'] = out['log_price'].diff()
        out['range_step'] = out['log_range'].diff()
        length = params.shape_bars
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        for lag in range(length - 1, -1, -1):
            dx = out['price_step'].shift(lag)
            dy = out['range_step'].shift(lag)
            area = area + 0.5 * (x * dy - y * dx)
            x = x + dx
            y = y + dy
        out['price_net'] = x
        out['range_net'] = y
        out['signature_area'] = area
        out['price_variation'] = out['price_step'].abs().rolling(length).sum()
        out['range_variation'] = out['range_step'].abs().rolling(length).sum()
        px = out['price_variation'].replace(0.0, np.nan)
        ry = out['range_variation'].replace(0.0, np.nan)
        out['normalized_area'] = area / (px * ry)
        out['price_efficiency'] = x / px
        out['range_efficiency'] = y / ry
        opening = data['open'].where(data['open'] > 0)
        out['matured_return'] = opening / opening.shift(params.horizon_days) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[['normalized_area', 'price_efficiency', 'range_efficiency']].to_numpy()
        realized = out['matured_return'].to_numpy()
        forecast = np.full(len(data), np.nan)
        counts = np.zeros(len(data), dtype=int)
        # ponytail: bounded O(n * history) scan; use indexed neighbors if histories grow substantially.
        for i in range(len(data)):
            if not np.isfinite(features[i]).all():
                continue
            first = max(length, i - params.history_bars)
            stop = i - params.horizon_days
            if stop <= first:
                continue
            candidates = np.arange(first, stop)
            # Candidate j enters at open j+1; its outcome matures at open j+1+horizon.
            outcomes = realized[candidates + params.horizon_days + 1]
            valid = np.isfinite(features[candidates]).all(axis=1) & np.isfinite(outcomes)
            candidates = candidates[valid]
            outcomes = outcomes[valid]
            if len(candidates) < params.neighbors:
                continue
            delta = features[candidates] - features[i]
            distance = 4.0 * delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2
            selected = np.argsort(distance, kind='stable')[:params.neighbors]
            forecast[i] = outcomes[selected].mean()
            counts[i] = len(selected)
        out['analog_forecast'] = forecast
        out['analog_count'] = counts
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        forecast = indicators['analog_forecast']
        bullish = np.isfinite(forecast) & forecast.gt(params.entry_edge)
        # Two consecutive bullish closes enter; the first false condition exits.
        confirmed = bullish & bullish.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df['signal'] = confirmed.astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = ShapeParams(shape_bars=4, history_bars=60, neighbors=8)
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=120, freq='B')
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.0005 * t + 0.015 * np.sin(t / 3.0))
    spread = 0.01 + 0.004 * np.sin(t / 5.0)
    data = pd.DataFrame({'open': close * np.exp(0.002 * np.cos(t)),
                         'high': close * (1.0 + spread),
                         'low': close * (1.0 - spread),
                         'close': close, 'volume': 1000000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:90], params)
    assert full['analog_forecast'].notna().any()
    assert np.allclose(full['analog_forecast'].iloc[:90],
                       prefix['analog_forecast'], equal_nan=True)
    sample = data.iloc[:7]
    evidence = pd.DataFrame({'analog_forecast': [np.nan, 0.01, 0.01, 0.01, -0.01, 0.01, 0.01]},
                            index=sample.index)
    signals = strategy.generate_signals(sample, evidence, None, params).data
    assert signals['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert signals['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= params.shape_bars + 1
