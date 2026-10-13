from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 4
    scale_bars: int = 20
    archive_bars: int = 252
    neighbors: int = 12
    hold_bars: int = 8
    min_edge: float = 0.002


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1790971387'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.scale_bars + params.shape_bars + params.archive_bars + params.hold_bars + 2)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        if min(p.shape_bars, p.scale_bars, p.archive_bars, p.neighbors, p.hold_bars) < 1:
            raise ValueError('Lookbacks, neighbors, and holding period must be positive.')
        if p.scale_bars < 2 or p.archive_bars < p.neighbors or not np.isfinite(p.min_edge) or p.min_edge <= 0:
            raise ValueError('Invalid analog parameters.')
        out = pd.DataFrame(index=data.index)
        prior = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = (data['open'].where(lambda x: x > 0) / prior - 1).replace([np.inf, -np.inf], np.nan)
        out['gap_scale'] = out['gap'].rolling(p.scale_bars, min_periods=p.scale_bars).std().shift(1)
        out['scaled_gap'] = (out['gap'] / out['gap_scale'].where(out['gap_scale'] > 1e-8)).clip(-6, 6)
        span = data['high'] - data['low']
        out['location'] = ((2 * data['close'] - data['high'] - data['low']) / span.where(span > 0)).clip(-1, 1)
        for lag in range(p.shape_bars):
            out[f'gap_{lag}'] = out['scaled_gap'].shift(lag)
            out[f'location_{lag}'] = out['location'].shift(lag)
        gap_shapes = out[[f'gap_{lag}' for lag in range(p.shape_bars)]].to_numpy(dtype=float)
        location_shapes = out[[f'location_{lag}' for lag in range(p.shape_bars)]].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        gap_forecast = np.full(len(data), np.nan)
        location_forecast = np.full(len(data), np.nan)
        start = p.scale_bars + p.shape_bars + p.archive_bars + p.hold_bars + 1
        # ponytail: bounded brute-force neighbors; batch distances if archives grow substantially.
        for t in range(start, len(data)):
            if not (np.isfinite(gap_shapes[t]).all() and np.isfinite(location_shapes[t]).all()):
                continue
            # A historical decision j fills at j+1 and exits at j+1+hold_bars.
            # Its exit open must already be observed at the current decision close.
            last = t - p.hold_bars - 1
            candidates = np.arange(max(0, last - p.archive_bars + 1), last + 1)
            entry = opens[candidates + 1]
            exit_price = opens[candidates + p.hold_bars + 1]
            valid = np.isfinite(entry) & np.isfinite(exit_price) & (entry > 0) & (exit_price > 0)
            valid &= np.isfinite(gap_shapes[candidates]).all(axis=1)
            valid &= np.isfinite(location_shapes[candidates]).all(axis=1)
            candidates = candidates[valid]
            if len(candidates) < p.neighbors:
                continue
            outcomes = np.log(opens[candidates + p.hold_bars + 1] / opens[candidates + 1])
            gap_distance = np.square(gap_shapes[candidates] - gap_shapes[t]).mean(axis=1)
            location_distance = np.square(location_shapes[candidates] - location_shapes[t]).mean(axis=1)
            gap_nearest = np.argsort(gap_distance, kind='stable')[:p.neighbors]
            location_nearest = np.argsort(location_distance, kind='stable')[:p.neighbors]
            gap_forecast[t] = outcomes[gap_nearest].mean()
            location_forecast[t] = outcomes[location_nearest].mean()
        out['gap_forecast'] = gap_forecast
        out['location_forecast'] = location_forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        gap = indicators['gap_forecast']
        location = indicators['location_forecast']
        valid = np.isfinite(gap) & np.isfinite(location)
        buy = valid & (gap > params.min_edge) & (location > params.min_edge)
        sell = valid & (gap < -params.min_edge) & (location < -params.min_edge)
        entries = np.select([buy, sell], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for t in range(len(data)):
            if position != 0 and t == exit_at:
                position = 0
                continue
            if position == 0 and entries[t] != 0:
                position = int(entries[t])
                exit_at = t + params.hold_bars
            signals[t] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    p = AnalogParams()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame(index=index)
    ind = pd.DataFrame({'gap_forecast': 0.01, 'location_forecast': 0.01}, index=index)
    ind.loc[index[1]:, ['gap_forecast', 'location_forecast']] = -0.01
    result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0, -1, -1]
    ind['location_forecast'] = -ind['gap_forecast']
    result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
    assert result['signal'].eq(0).all()
