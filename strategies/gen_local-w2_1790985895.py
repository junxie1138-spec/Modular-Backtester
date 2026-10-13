from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    frac_order: float = 0.45
    frac_window: int = 32
    baseline_window: int = 40
    fast_range: int = 5
    slow_range: int = 20
    compression_limit: float = 0.75
    recapture_z: float = -0.5
    holding_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local-w2_1790985895'

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.frac_window + params.baseline_window + 1,
                       params.fast_range, params.slow_range))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        close = data['close'].astype(float)
        fractional_close = pd.Series(0.0, index=data.index)
        for k in range(params.frac_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        out['fractional_close'] = fractional_close
        out['fractional_mean'] = fractional_close.rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean().shift(1)
        out['fractional_std'] = fractional_close.rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).std(ddof=0).shift(1)
        out['fractional_z'] = (
            (fractional_close - out['fractional_mean'])
            / out['fractional_std'].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['previous_z'] = out['fractional_z'].shift(1)
        out['daily_range'] = (
            (data['high'] - data['low']) / close.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['fast_range'] = out['daily_range'].rolling(
            params.fast_range, min_periods=params.fast_range
        ).mean()
        out['slow_range'] = out['daily_range'].rolling(
            params.slow_range, min_periods=params.slow_range
        ).mean()
        out['compression_ratio'] = (
            out['fast_range'] / out['slow_range'].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['recovery_strength'] = (
            (out['fractional_z'] - out['previous_z']).clip(lower=0.0)
            / (1.0 + (out['fractional_z'] - out['previous_z']).clip(lower=0.0))
        )
        out['compression_depth'] = (
            1.0 - out['compression_ratio']
        ).clip(lower=0.0, upper=1.0)
        out['conviction'] = (
            out['recovery_strength'] * out['compression_depth']
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError('sizes must satisfy 0 < min_size <= max_size <= 1')
        ready = indicators[
            ['fractional_z', 'previous_z', 'compression_ratio']
        ].notna().all(axis=1)
        entry = (
            ready
            & (indicators['previous_z'] <= params.recapture_z)
            & (indicators['fractional_z'] > params.recapture_z)
            & (indicators['fractional_z'] < 0.75)
            & (indicators['compression_ratio'] < params.compression_limit)
        ).fillna(False).to_numpy(dtype=bool)
        proposed_size = (
            params.min_size
            + (params.max_size - params.min_size) * indicators['conviction']
        ).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                    sizes[i] = held_size
                else:
                    exit_bar = -1
                continue
            if entry[i]:
                held_size = float(proposed_size[i])
                signals[i] = 1
                sizes[i] = held_size
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
