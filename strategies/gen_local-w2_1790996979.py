from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryAccelerationParams:
    peak_window: int = 40
    trough_window: int = 15
    roc_window: int = 3
    volatility_window: int = 20
    minimum_depth: float = 0.025
    acceleration_threshold: float = 0.35
    hold_bars: int = 8
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryAccelerationParams]):
    strategy_id = 'gen_local-w2_1790996979'

    @classmethod
    def params_type(cls) -> type[RecoveryAccelerationParams]:
        return RecoveryAccelerationParams

    def warmup_bars(self, params: RecoveryAccelerationParams) -> int:
        return int(max(
            params.peak_window,
            params.trough_window,
            params.volatility_window + 1,
            2 * params.roc_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryAccelerationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['trough'] = close.rolling(params.trough_window, min_periods=params.trough_window).min()
        out['daily_return'] = close.pct_change(fill_method=None)
        out['volatility'] = out['daily_return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out['roc'] = close.pct_change(periods=params.roc_window, fill_method=None)
        out['acceleration'] = out['roc'] - out['roc'].shift(params.roc_window)
        out['normalizer'] = (
            out['volatility'] * np.sqrt(float(params.roc_window))
        ).where(out['volatility'] > 1e-12)
        out['acceleration_score'] = out['acceleration'] / out['normalizer']
        out['previous_acceleration_score'] = out['acceleration_score'].shift(1)
        out['drawdown_span'] = out['peak'] - out['trough']
        out['depth'] = out['drawdown_span'] / out['peak']
        out['recovery_fraction'] = (
            (close - out['trough']) / out['drawdown_span'].where(out['drawdown_span'] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryAccelerationParams,
    ) -> SignalFrame:
        score = indicators['acceleration_score']
        previous = indicators['previous_acceleration_score']
        recovery = indicators['recovery_fraction']
        valid = indicators[[
            'acceleration_score', 'previous_acceleration_score',
            'depth', 'recovery_fraction',
        ]].notna().all(axis=1)
        eligible = valid & (indicators['depth'] >= params.minimum_depth)
        threshold = max(float(params.acceleration_threshold), 1e-6)
        long_entry = (
            eligible
            & recovery.between(0.15, 0.55)
            & (score > threshold)
            & (previous <= threshold)
        )
        short_entry = (
            eligible
            & recovery.between(0.55, 0.85)
            & (score < -threshold)
            & (previous >= -threshold)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        minimum_size = max(float(params.minimum_size), 1e-6)
        maximum_size = max(float(params.maximum_size), minimum_size)
        strength = (score.abs() / threshold - 1.0).clip(lower=0.0, upper=2.0) / 2.0
        entry_sizes = (
            minimum_size + (maximum_size - minimum_size) * strength.fillna(0.0)
        ).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum_size, dtype=float)
        hold_bars = max(1, int(params.hold_bars))
        position = 0
        entry_bar = -1
        position_size = minimum_size
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar >= hold_bars:
                    position = 0
                    position_size = minimum_size
                    continue
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
                position_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
