from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureAreaParams:
    path_bars: int = 4
    volatility_bars: int = 20
    area_threshold: float = 0.12
    stretch_threshold: float = 1.0
    holding_bars: int = 2
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SignatureAreaParams]):
    strategy_id = 'gen_local-w2_1790991466'

    @classmethod
    def params_type(cls) -> type[SignatureAreaParams]:
        return SignatureAreaParams

    def warmup_bars(self, params: SignatureAreaParams) -> int:
        return max(2, int(params.volatility_bars)) + max(2, int(params.path_bars)) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureAreaParams) -> pd.DataFrame:
        length = max(2, int(params.path_bars))
        baseline = max(2, int(params.volatility_bars))
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        volume = data['volume'].astype(float).where(data['volume'] >= 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['return'].rolling(baseline, min_periods=baseline).std(ddof=0)
        out['volume_baseline'] = volume.rolling(baseline, min_periods=baseline).mean()
        out['volume_clock_step'] = (volume / out['volume_baseline'].where(out['volume_baseline'] > 0)).replace([np.inf, -np.inf], np.nan)

        # Antisymmetric second-order signature of (return, volume-clock) increments.
        area = pd.Series(0.0, index=data.index)
        for earlier in range(1, length):
            for later in range(earlier):
                area = area + 0.5 * (
                    out['return'].shift(earlier) * out['volume_clock_step'].shift(later)
                    - out['volume_clock_step'].shift(earlier) * out['return'].shift(later)
                )
        out['signature_area'] = area
        out['absolute_path'] = out['return'].abs().rolling(length, min_periods=length).sum()
        out['clock_length'] = out['volume_clock_step'].rolling(length, min_periods=length).sum()
        denominator = out['absolute_path'] * out['clock_length']
        out['normalized_area'] = (area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        out['displacement'] = out['return'].rolling(length, min_periods=length).sum()
        scale = out['volatility'] * np.sqrt(float(length))
        out['stretch'] = (out['displacement'] / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureAreaParams) -> SignalFrame:
        area_threshold = max(1e-6, float(params.area_threshold))
        stretch_threshold = max(1e-6, float(params.stretch_threshold))
        holding = min(2, max(1, int(params.holding_bars)))
        minimum_size = min(1.0, max(0.01, float(params.minimum_size)))
        maximum_size = min(1.0, max(minimum_size, float(params.maximum_size)))
        area = indicators['normalized_area']
        stretch = indicators['stretch']
        valid = pd.Series(
            np.isfinite(area.to_numpy()) & np.isfinite(stretch.to_numpy()),
            index=data.index,
        )
        qualified = (
            valid
            & (area.abs() >= area_threshold)
            & (stretch.abs() >= stretch_threshold)
            & (area * stretch < 0)
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[qualified & (area > 0)] = 1
        candidates.loc[qualified & (area < 0)] = -1

        # Exposure increases with both area and displacement; freeze it at entry.
        strength = np.minimum(area.abs() / area_threshold, stretch.abs() / stretch_threshold)
        strength = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        proposed_size = minimum_size + (maximum_size - minimum_size) * (strength / (1.0 + strength))
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum_size, dtype=float)
        candidate_values = candidates.to_numpy()
        size_values = proposed_size.to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_size = minimum_size
        for i in range(len(data)):
            if position != 0 and i - entry_bar >= holding:
                # A flat bar is mandatory at expiry; entries cannot override it.
                position = 0
                entry_size = minimum_size
            elif position == 0 and candidate_values[i] != 0:
                position = int(candidate_values[i])
                entry_bar = i
                entry_size = float(size_values[i])
            signals[i] = position
            sizes[i] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
