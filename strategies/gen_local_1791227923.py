from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 16
    percentile_bars: int = 126
    entry_percentile: float = 0.80
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local_1791227923'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.graph_bars + params.percentile_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if not (params.graph_bars >= 2 and params.percentile_bars >= 2 and params.atr_bars >= 1 and 0.0 < params.entry_percentile < 1.0 and params.trail_atr > 0.0 and params.max_hold_bars >= 1):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['log_close'] = np.log(close.where(close > 0.0))
        log_close = out['log_close']
        slopes = np.column_stack([
            ((log_close - log_close.shift(distance)) / distance).to_numpy()
            for distance in range(1, params.graph_bars + 1)
        ])
        valid = np.isfinite(slopes).all(axis=1)
        lower_bound = slopes[:, 0].copy()
        upper_bound = slopes[:, 0].copy()
        upright = np.ones(len(data), dtype=float)
        inverted = np.ones(len(data), dtype=float)
        # A distant endpoint is visible when every intervening close is below
        # its connecting line; inversion swaps below for above.
        for distance in range(1, params.graph_bars):
            slope = slopes[:, distance]
            upright += slope < lower_bound
            inverted += slope > upper_bound
            lower_bound = np.minimum(lower_bound, slope)
            upper_bound = np.maximum(upper_bound, slope)
        out['upright_degree'] = np.where(valid, upright, np.nan)
        out['inverted_degree'] = np.where(valid, inverted, np.nan)
        out['topology'] = (out['upright_degree'] - out['inverted_degree']) / (out['upright_degree'] + out['inverted_degree'])
        out['topology_threshold'] = out['topology'].abs().shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.entry_percentile)
        out['displacement'] = log_close - log_close.shift(params.graph_bars)
        previous_close = close.shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        topology = indicators['topology'].to_numpy(dtype=float)
        threshold = indicators['topology_threshold'].to_numpy(dtype=float)
        displacement = indicators['displacement'].to_numpy(dtype=float)
        ready = np.isfinite(close) & (close > 0.0) & np.isfinite(atr) & (atr > 0.0) & np.isfinite(topology) & np.isfinite(threshold) & np.isfinite(displacement)
        extreme = np.abs(topology) > threshold
        entry = np.zeros(len(data), dtype=int)
        entry[ready & extreme & (topology > 0.0) & (displacement > 0.0)] = 1
        entry[ready & extreme & (topology < 0.0) & (displacement < 0.0)] = -1
        signal = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                if not ready[i]:
                    position = 0
                else:
                    age += 1
                    # First held close is on the bar after the entry decision.
                    if np.isnan(watermark):
                        watermark = close[i]
                        stop = close[i] - position * params.trail_atr * atr[i]
                    elif position == 1:
                        watermark = max(watermark, close[i])
                        stop = max(stop, watermark - params.trail_atr * atr[i])
                    else:
                        watermark = min(watermark, close[i])
                        stop = min(stop, watermark + params.trail_atr * atr[i])
                    stopped = close[i] <= stop if position == 1 else close[i] >= stop
                    if stopped or age >= params.max_hold_bars:
                        position = 0
            elif entry[i]:
                position = int(entry[i])
                age = 0
                watermark = np.nan
                stop = np.nan
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': np.ones(len(data), dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = VisibilityParams(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=5)
    for direction, closes in [(1, [100.0, 101.0, 110.0, 106.0, 106.0]), (-1, [100.0, 99.0, 90.0, 94.0, 94.0])]:
        data = pd.DataFrame({'open': closes, 'high': closes, 'low': closes, 'close': closes, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 3.0, 3.0], 'topology': [float(direction), 0.0, 0.0, 0.0, 0.0], 'topology_threshold': 0.5, 'displacement': float(direction)}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, derived, None, params)
        assert result.data['signal'].tolist() == [0, direction, direction, direction, 0]
        assert (result.data['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
