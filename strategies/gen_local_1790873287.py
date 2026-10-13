from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    path_window: int = 5
    area_threshold: float = -0.08
    downside_recent: int = 3
    downside_baseline: int = 12
    contraction_threshold: float = 0.65
    volatility_window: int = 20
    target_daily_volatility: float = 0.006
    atr_window: int = 14
    breakeven_gain: float = 0.006
    trail_atr: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790873287'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            max(2, params.path_window) + 1,
            max(2, params.volatility_window) + 1,
            max(1, params.atr_window) + 1,
            max(1, params.downside_recent)
            + max(2, params.downside_baseline) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        n = max(2, params.path_window)
        recent = max(1, params.downside_recent)
        baseline = max(2, params.downside_baseline)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].clip(lower=0)
        out['volume_mean'] = volume.rolling(n, min_periods=n).mean()
        # Constant scaling preserves the normalized signature area.
        out['volume_increment'] = volume / 1000000.0
        r = out['return']
        v = out['volume_increment']
        area = pd.Series(0.0, index=data.index)
        for older in range(1, n):
            for newer in range(older):
                area = area + 0.5 * (
                    r.shift(older) * v.shift(newer)
                    - v.shift(older) * r.shift(newer)
                )
        out['signature_area_raw'] = area
        out['signature_scale'] = (
            r.abs().rolling(n, min_periods=n).sum()
            * v.rolling(n, min_periods=n).sum()
        )
        out['signature_area'] = area / out['signature_scale'].where(out['signature_scale'] > 0)
        out['downside_squared'] = r.clip(upper=0).pow(2)
        out['downside_recent'] = out['downside_squared'].rolling(recent, min_periods=recent).mean()
        out['downside_baseline'] = (
            out['downside_squared'].shift(recent)
            .rolling(baseline, min_periods=baseline).mean()
        )
        out['downside_ratio'] = (
            out['downside_recent']
            / out['downside_baseline'].where(out['downside_baseline'] > 0)
        )
        vw = max(2, params.volatility_window)
        out['realized_volatility'] = r.rolling(vw, min_periods=vw).std(ddof=0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        aw = max(1, params.atr_window)
        out['atr'] = out['true_range'].rolling(aw, min_periods=aw).mean()
        out['target_size'] = (
            max(0.000001, params.target_daily_volatility)
            / out['realized_volatility'].where(out['realized_volatility'] > 0)
        ).clip(lower=0.01, upper=1.0).fillna(0.01)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        qualified = (
            indicators['signature_area'].lt(params.area_threshold)
            & indicators['downside_ratio'].lt(params.contraction_threshold)
            & indicators['atr'].gt(0)
            & indicators['realized_volatility'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        target_sizes = indicators['target_size'].fillna(0.01).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        active = False
        pending = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        armed = False
        age = 0
        held_size = 0.01
        multiplier = max(0.01, params.trail_atr)
        activation = max(0.0, params.breakeven_gain)
        horizon = min(2, max(1, params.max_hold_bars))
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atrs[i - 1]):
                    active = True
                    entry = opens[i]
                    peak = entry
                    stop = max(0.0, entry - multiplier * atrs[i - 1])
                    armed = False
                    age = 0
            if active:
                age += 1
                sizes[i] = held_size
                valid = (
                    np.isfinite(highs[i]) and np.isfinite(lows[i])
                    and np.isfinite(closes[i]) and np.isfinite(atrs[i])
                )
                if not valid or lows[i] <= stop or age >= horizon:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if peak >= entry * (1.0 + activation):
                    armed = True
                if armed:
                    # Ratchets computed at this close apply from the next bar.
                    stop = max(stop, entry, peak - multiplier * atrs[i])
                if closes[i] <= stop:
                    active = False
                    continue
                signals[i] = 1
            elif i >= warmup - 1 and qualified[i]:
                held_size = float(target_sizes[i])
                sizes[i] = held_size
                signals[i] = 1
                pending = True
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
