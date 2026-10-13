from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    half_window: int = 4
    coherence_window: int = 5
    drawdown_window: int = 30
    percentile_window: int = 100
    entry_quantile: float = 0.8
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1790998975'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        h = max(2, int(params.half_window))
        c = max(2, int(params.coherence_window))
        d = max(2, int(params.drawdown_window))
        r = max(2, int(params.percentile_window))
        a = max(2, int(params.atr_window))
        return int(max(2 * h + c + r, d + r + 10, a + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        h = max(2, int(params.half_window))
        c = max(2, int(params.coherence_window))
        d = max(2, int(params.drawdown_window))
        r = max(2, int(params.percentile_window))
        a = max(2, int(params.atr_window))
        q = float(np.clip(params.entry_quantile, 0.55, 0.95))
        close = data['close'].astype(float).where(data['close'] > 0)
        log_close = np.log(close)
        recent_level = log_close.rolling(h, min_periods=h).mean()
        older_level = recent_level.shift(h)
        slope = (recent_level - older_level) / h
        innovation = slope.diff()
        innovation_net = innovation.rolling(c, min_periods=c).sum()
        innovation_activity = innovation.abs().rolling(c, min_periods=c).sum()
        coherence = innovation_net / innovation_activity.replace(0.0, np.nan)
        drawdown = 1.0 - close / close.rolling(d, min_periods=d).max()
        depth_threshold = drawdown.shift(1).rolling(r, min_periods=r).quantile(q)
        upper_coherence = coherence.shift(1).rolling(r, min_periods=r).quantile(q)
        lower_coherence = coherence.shift(1).rolling(r, min_periods=r).quantile(1.0 - q)
        deep_drawdown = (drawdown > depth_threshold).where(depth_threshold.notna()).astype(float)
        recent_deep = deep_drawdown.shift(1).rolling(10, min_periods=10).max()
        prior_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - prior_close).abs(),
            (data['low'] - prior_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(prior_close.notna())
        atr = true_range.rolling(a, min_periods=a).mean()
        return pd.DataFrame({
            'slope': slope,
            'innovation': innovation,
            'innovation_net': innovation_net,
            'innovation_activity': innovation_activity,
            'coherence': coherence,
            'drawdown': drawdown,
            'depth_threshold': depth_threshold,
            'upper_coherence': upper_coherence,
            'lower_coherence': lower_coherence,
            'recent_deep': recent_deep,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        valid = indicators[['slope', 'coherence', 'drawdown', 'depth_threshold',
                            'upper_coherence', 'lower_coherence', 'recent_deep',
                            'atr']].notna().all(axis=1)
        valid &= indicators['atr'] > 0
        long_entry = (
            valid
            & (indicators['drawdown'] > indicators['depth_threshold'])
            & (indicators['slope'] < 0)
            & (indicators['coherence'] > 0)
            & (indicators['coherence'] > indicators['upper_coherence'])
        )
        short_entry = (
            valid
            & (indicators['drawdown'] > 0)
            & (indicators['recent_deep'] > 0)
            & (indicators['slope'] > 0)
            & (indicators['coherence'] < 0)
            & (indicators['coherence'] < indicators['lower_coherence'])
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        side = 0
        entry = np.nan
        stop = np.nan
        pending_atr = np.nan
        activated = False
        age = 0
        k = max(0.1, float(params.trail_atr))
        trigger = max(0.0, float(params.breakeven_pct))
        max_hold = max(1, int(params.max_hold_bars))
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            price = closes[i]
            if side != 0:
                if not np.isfinite(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        entry = np.nan
                        continue
                    stop = entry - side * k * pending_atr
                    age = 0
                age += 1
                if not np.isfinite(price) or price <= 0:
                    signals[i] = side
                    continue
                breached = price <= stop if side == 1 else price >= stop
                if breached or age >= max_hold:
                    side = 0
                    entry = np.nan
                    activated = False
                    continue
                if side * (price / entry - 1.0) >= trigger:
                    activated = True
                if activated:
                    if side == 1:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, price - k * atrs[i])
                    else:
                        stop = min(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = min(stop, price + k * atrs[i])
                signals[i] = side
            elif i >= warmup - 1 and candidates[i] != 0:
                if np.isfinite(price) and price > 0:
                    side = int(candidates[i])
                    entry = np.nan
                    pending_atr = atrs[i]
                    activated = False
                    signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
