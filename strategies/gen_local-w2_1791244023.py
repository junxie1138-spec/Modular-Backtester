from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    seller_window: int = 6
    percentile_window: int = 63
    drawdown_quantile: float = 0.25
    buying_quantile: float = 0.80
    atr_window: int = 14
    breakeven_gain: float = 0.012
    trail_atr: float = 1.8
    max_hold: int = 5
    cooldown_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791244023"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(params.peak_window, params.seller_window, params.atr_window) + params.percentile_window + 3)

    def indicators(self, data, params):
        close = data['close']
        previous_close = close.shift(1)
        volume = data['volume'].clip(lower=0)
        change = close.diff()
        drawdown = close / close.rolling(params.peak_window).max().replace(0, np.nan) - 1.0
        selling_volume = volume.where(change.lt(0), 0.0)
        typical_price = (data['high'] + data['low'] + close) / 3.0
        seller_basis = (typical_price * selling_volume).rolling(params.seller_window).sum() / selling_volume.rolling(params.seller_window).sum().replace(0, np.nan)
        span = (data['high'] - data['low']).replace(0, np.nan)
        closing_pressure = ((close - data['low']) / span).clip(0.0, 1.0)
        buying_participation = volume * closing_pressure * change.gt(0).astype(float)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out['drawdown'] = drawdown
        out['drawdown_cutoff'] = drawdown.rolling(params.percentile_window).quantile(params.drawdown_quantile).shift(1)
        out['seller_basis'] = seller_basis.shift(1)
        out['buying_participation'] = buying_participation
        out['buying_cutoff'] = buying_participation.rolling(params.percentile_window).quantile(params.buying_quantile).shift(1)
        out['atr'] = true_range.rolling(params.atr_window).mean()
        out['entry'] = (
            drawdown.shift(1).lt(out['drawdown_cutoff'])
            & close.gt(out['seller_basis'])
            & previous_close.le(out['seller_basis'].shift(1))
            & change.gt(0)
            & buying_participation.gt(out['buying_cutoff'])
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        entry_bar = 0
        next_eligible = 0
        for i in range(count):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]):
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    stop = entry_price - params.trail_atr * atrs[i - 1]
            if active:
                close = closes[i]
                expired = i - entry_bar + 1 >= params.max_hold
                breached = np.isfinite(close) and close <= stop
                if expired or breached:
                    active = False
                    next_eligible = i + params.cooldown_bars + 1
                else:
                    if np.isfinite(close):
                        armed = armed or close >= entry_price * (1.0 + params.breakeven_gain)
                        if armed:
                            stop = max(stop, entry_price)
                            if np.isfinite(atrs[i]) and atrs[i] > 0:
                                stop = max(stop, close - params.trail_atr * atrs[i])
                    signals[i] = 1
            elif i >= next_eligible and i + 1 < count and entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': np.full(count, 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
