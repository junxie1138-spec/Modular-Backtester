from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 12
    memory_window: int = 40
    recurrence_lag: int = 5
    atr_window: int = 14
    correlation_floor: float = 0.15
    entry_strength: float = 0.12
    atr_multiple: float = 3.0
    holding_bars: int = 18
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1790985103'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.slope_window + params.memory_window + params.recurrence_lag + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        n = params.slope_window
        lag = params.recurrence_lag
        close = data['close'].astype(float)
        log_close = np.log(close.where(close > 0))
        coordinate = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        price_sum = log_close.rolling(n, min_periods=n).sum()
        coordinate_mean = coordinate.rolling(n, min_periods=n).mean()
        weighted_sum = (coordinate * log_close).rolling(n, min_periods=n).sum()
        denominator = n * (n * n - 1) / 12.0
        slope = (weighted_sum - coordinate_mean * price_sum) / denominator
        innovation = slope.diff()
        delayed_innovation = innovation.shift(lag)
        correlation = innovation.rolling(
            params.memory_window, min_periods=params.memory_window
        ).corr(delayed_innovation).shift(1)
        dispersion = innovation.rolling(
            params.memory_window, min_periods=params.memory_window
        ).std(ddof=0).shift(1)
        forecast = correlation * innovation.shift(lag - 1)
        strength = forecast / dispersion.where(dispersion > 1e-12)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        result = pd.DataFrame({
            'log_close': log_close,
            'coordinate': coordinate,
            'price_sum': price_sum,
            'coordinate_mean': coordinate_mean,
            'weighted_sum': weighted_sum,
            'slope': slope,
            'innovation': innovation,
            'delayed_innovation': delayed_innovation,
            'correlation': correlation,
            'dispersion': dispersion,
            'forecast': forecast,
            'strength': strength,
            'previous_close': previous_close,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        ready = indicators[['slope', 'correlation', 'strength', 'atr']].notna().all(axis=1)
        entry = (
            ready
            & (indicators['slope'] > 0)
            & (indicators['correlation'] >= params.correlation_floor)
            & (indicators['strength'] >= params.entry_strength)
            & (indicators['atr'] > 0)
        ).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        active = False
        high_water = -np.inf
        trailing_stop = -np.inf
        age = 0
        entry_size = params.minimum_size
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if active:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                else:
                    # The first active iteration is the execution bar following entry intent.
                    high_water = max(high_water, close[i])
                    trailing_stop = max(
                        trailing_stop,
                        high_water - params.atr_multiple * atr[i],
                    )
                    if close[i] <= trailing_stop or age >= params.holding_bars:
                        active = False
                if active:
                    signals[i] = 1
                    sizes[i] = entry_size
            elif i >= warmup - 1 and entry[i] and np.isfinite(close[i]) and close[i] > 0:
                active = True
                age = 0
                high_water = -np.inf
                trailing_stop = -np.inf
                confidence = float(np.clip(strength[i] / 1.5, 0.0, 1.0))
                entry_size = float(
                    params.minimum_size
                    + (params.maximum_size - params.minimum_size) * confidence
                )
                signals[i] = 1
                sizes[i] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
