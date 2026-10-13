from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    recent_bars: int = 3
    baseline_bars: int = 24
    atr_bars: int = 14
    change_threshold: float = 1.8
    volume_ratio: float = 1.2
    directional_volume: float = 0.45
    breakeven_pct: float = 0.008
    initial_stop_atr: float = 1.6
    trail_atr: float = 1.4
    holding_bars: int = 4
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = 'gen_local-w2_1791000717'

    @classmethod
    def params_type(cls) -> type[ChangepointParams]:
        return ChangepointParams

    def warmup_bars(self, params: ChangepointParams) -> int:
        recent = max(2, int(params.recent_bars))
        baseline = max(3, int(params.baseline_bars))
        return int(max(recent + baseline + 1, max(2, int(params.atr_bars)) + 1))

    def indicators(self, data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        recent = max(2, int(params.recent_bars))
        baseline = max(3, int(params.baseline_bars))
        atr_bars = max(2, int(params.atr_bars))
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        volume = data['volume'].astype(float).clip(lower=0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volume'] = volume
        out['return_volume'] = out['return'] * volume
        out['signed_volume'] = np.sign(out['return']) * volume
        out['recent_volume_sum'] = volume.rolling(recent, min_periods=recent).sum()
        out['baseline_volume_sum'] = volume.shift(recent).rolling(baseline, min_periods=baseline).sum()
        out['recent_mean'] = out['return_volume'].rolling(recent, min_periods=recent).sum() / out['recent_volume_sum'].replace(0, np.nan)
        out['baseline_mean'] = out['return_volume'].shift(recent).rolling(baseline, min_periods=baseline).sum() / out['baseline_volume_sum'].replace(0, np.nan)
        out['baseline_std'] = out['return'].shift(recent).rolling(baseline, min_periods=baseline).std(ddof=0)
        out['mean_shift'] = out['recent_mean'] - out['baseline_mean']
        out['shift_score'] = out['mean_shift'] / (out['baseline_std'].replace(0, np.nan) * np.sqrt(1.0 / recent + 1.0 / baseline))
        out['volume_ratio'] = (out['recent_volume_sum'] / recent) / (out['baseline_volume_sum'] / baseline).replace(0, np.nan)
        out['volume_balance'] = out['signed_volume'].rolling(recent, min_periods=recent).sum() / out['recent_volume_sum'].replace(0, np.nan)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_bars, min_periods=atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ChangepointParams) -> SignalFrame:
        ind = indicators
        threshold = max(0.1, float(params.change_threshold))
        volume_threshold = max(0.1, float(params.volume_ratio))
        balance_threshold = float(np.clip(params.directional_volume, 0.01, 1.0))
        minimum_size = float(np.clip(params.minimum_size, 0.01, 1.0))
        horizon = int(np.clip(params.holding_bars, 3, 5))
        ready = ind[['shift_score', 'volume_ratio', 'volume_balance', 'atr']].notna().all(axis=1) & ind['atr'].gt(0)
        ready &= pd.Series(np.arange(len(data)) >= self.warmup_bars(params) - 1, index=data.index)
        direction = np.sign(ind['shift_score']).fillna(0)
        qualifies = ready & ind['shift_score'].abs().ge(threshold)
        qualifies &= ind['volume_ratio'].ge(volume_threshold)
        qualifies &= (direction * ind['volume_balance']).ge(balance_threshold)
        qualifies &= (direction * ind['recent_mean']).gt(0)
        qualified_side = direction.where(qualifies, 0).astype(int)
        onset = qualified_side.ne(0) & qualified_side.ne(qualified_side.shift(1).fillna(0))
        candidates = qualified_side.where(onset, 0).to_numpy(dtype=int)
        strength = (
            (ind['shift_score'].abs() / threshold - 1.0).clip(0, 1)
            + (ind['volume_ratio'] / volume_threshold - 1.0).clip(0, 1)
            + ((ind['volume_balance'].abs() - balance_threshold) / max(0.01, 1.0 - balance_threshold)).clip(0, 1)
        ) / 3.0
        conviction = (minimum_size + (1.0 - minimum_size) * strength).fillna(minimum_size).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = ind['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum_size, dtype=float)
        side = 0
        pending_side = 0
        pending_size = minimum_size
        position_size = minimum_size
        entry = 0.0
        stop = 0.0
        age = 0
        armed = False
        for i in range(len(data)):
            # A decision made yesterday enters at today's opening price.
            if pending_side:
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atrs[i - 1]):
                    side = pending_side
                    entry = opens[i]
                    position_size = pending_size
                    # Signed prices make the protective stop ratchet upward for either direction.
                    stop = side * entry - max(0.1, params.initial_stop_atr) * atrs[i - 1]
                    age = 0
                    armed = False
                pending_side = 0
            if side:
                age += 1
                signed_close = side * closes[i]
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                exit_now = not valid_close or signed_close <= stop or age >= horizon
                if not exit_now:
                    gain = side * (closes[i] / entry - 1.0)
                    if gain >= max(0.0001, params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, signed_close - max(0.1, params.trail_atr) * atrs[i])
                        exit_now = signed_close <= stop
                if exit_now:
                    side = 0
                    # Exit decisions execute next bar; do not replace them with an entry.
                    continue
                signals[i] = side
                sizes[i] = position_size
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending_side = int(candidates[i])
                pending_size = float(conviction[i])
                signals[i] = pending_side
                sizes[i] = pending_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
