from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeReplacementParams:
    peak_window: int = 20
    volume_window: int = 20
    participation_window: int = 5
    minimum_drawdown: float = 0.015
    majority_fraction: float = 0.65
    relative_volume_min: float = 1.05
    rebound_fraction_min: float = 0.15
    profit_target: float = 0.008
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[VolumeReplacementParams]):
    strategy_id = 'gen_local-w2_1790998387'

    @classmethod
    def params_type(cls):
        return VolumeReplacementParams

    def warmup_bars(self, params: VolumeReplacementParams) -> int:
        return int(max(1, params.peak_window, params.volume_window) + max(1, params.participation_window) + 3)

    def indicators(self, data: pd.DataFrame, params: VolumeReplacementParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        volume = data['volume'].astype(float).clip(lower=0.0)
        peak_window = max(1, int(params.peak_window))
        volume_window = max(1, int(params.volume_window))
        participation_window = max(1, int(params.participation_window))
        out['close_change'] = close.diff()
        out['prior_peak'] = data['high'].rolling(peak_window, min_periods=peak_window).max().shift(1)
        out['drawdown'] = (close / out['prior_peak'].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['prior_volume_mean'] = volume.rolling(volume_window, min_periods=volume_window).mean().shift(1)
        out['relative_volume'] = (volume / out['prior_volume_mean'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        valid = out[['close_change', 'drawdown', 'relative_volume']].notna().all(axis=1) & close.gt(0.0) & volume.notna()
        out['underwater'] = valid & out['drawdown'].le(-abs(params.minimum_drawdown))
        out['underwater_volume'] = volume.where(out['underwater'], 0.0).where(valid)
        out['advancing_volume'] = volume.where(out['underwater'] & out['close_change'].gt(0.0), 0.0).where(valid)
        out['declining_volume'] = volume.where(out['underwater'] & out['close_change'].lt(0.0), 0.0).where(valid)
        out['volume_pool'] = out['underwater_volume'].rolling(participation_window, min_periods=participation_window).sum()
        out['advance_pool'] = out['advancing_volume'].rolling(participation_window, min_periods=participation_window).sum()
        out['decline_pool'] = out['declining_volume'].rolling(participation_window, min_periods=participation_window).sum()
        denominator = out['volume_pool'].replace(0.0, np.nan)
        out['advance_fraction'] = out['advance_pool'] / denominator
        out['decline_fraction'] = out['decline_pool'] / denominator
        active = out['underwater'] & out['relative_volume'].ge(params.relative_volume_min)
        out['recovery_bar'] = (active & out['close_change'].gt(0.0) & close.gt(data['open']) & out['advance_fraction'].ge(params.majority_fraction)).fillna(False)
        out['failure_bar'] = (active & out['close_change'].lt(0.0) & close.lt(data['open']) & out['decline_fraction'].ge(params.majority_fraction) & out['advance_fraction'].ge(params.rebound_fraction_min)).fillna(False)
        out['confirmed_recovery'] = out['recovery_bar'] & out['recovery_bar'].shift(1, fill_value=False)
        out['confirmed_failure'] = out['failure_bar'] & out['failure_bar'].shift(1, fill_value=False)
        out['entry_direction'] = np.select([out['confirmed_recovery'], out['confirmed_failure']], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolumeReplacementParams) -> SignalFrame:
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = np.nan
        age = 0
        holding_bars = min(2, max(1, int(params.holding_bars)))
        target = max(0.000001, float(params.profit_target))
        for i in range(len(data)):
            if side:
                # A decision from the previous close fills at this open.
                if not np.isfinite(entry_price):
                    entry_price = opens[i]
                    age = 0
                age += 1
                if not np.isfinite(entry_price) or entry_price <= 0.0 or not np.isfinite(closes[i]):
                    side = 0
                    entry_price = np.nan
                elif side * (closes[i] / entry_price - 1.0) >= target or age >= holding_bars:
                    # Exit decisions execute at the following open.
                    side = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(entries[i])
                entry_price = np.nan
                age = 0
            raw[i] = side
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')
