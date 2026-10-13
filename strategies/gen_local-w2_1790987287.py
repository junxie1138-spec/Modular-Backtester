from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DisplacementParams:
    ma_bars: int = 30
    regime_bars: int = 40
    atr_bars: int = 14
    efficiency_threshold: float = 0.30
    z_inner: float = 0.25
    z_outer: float = 1.50
    stop_atr: float = 2.50
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[DisplacementParams]):
    strategy_id = 'gen_local-w2_1790987287'

    @classmethod
    def params_type(cls) -> type[DisplacementParams]:
        return DisplacementParams

    def warmup_bars(self, params: DisplacementParams) -> int:
        return int(max(max(2, params.ma_bars) + 2,
                       max(2, params.regime_bars) + 2,
                       max(1, params.atr_bars) + 2))

    def indicators(self, data: pd.DataFrame, params: DisplacementParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        ma_bars = max(2, int(params.ma_bars))
        regime_bars = max(2, int(params.regime_bars))
        atr_bars = max(1, int(params.atr_bars))
        ma = close.rolling(ma_bars, min_periods=ma_bars).mean()
        dispersion = close.rolling(ma_bars, min_periods=ma_bars).std(ddof=0)
        z = (close - ma) / dispersion.where(dispersion > 0.0)
        step = close.diff()
        travel = step.abs().rolling(regime_bars, min_periods=regime_bars).sum()
        displacement = close.diff(regime_bars)
        efficiency = displacement.abs() / travel.where(travel > 0.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(atr_bars, min_periods=atr_bars).mean()
        result = pd.DataFrame({
            'ma': ma,
            'z': z,
            'z_change': z.diff(),
            'efficiency': efficiency,
            'regime_displacement': displacement,
            'atr': atr
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DisplacementParams) -> SignalFrame:
        z = indicators['z']
        efficiency = indicators['efficiency']
        recovering = indicators['z_change'] > 0.0
        inner = max(0.0, float(params.z_inner))
        outer = max(inner, float(params.z_outer))
        threshold = float(params.efficiency_threshold)
        trend_candidate = (
            (efficiency >= threshold)
            & (indicators['regime_displacement'] > 0.0)
            & z.between(inner, outer)
            & recovering
        )
        congestion_candidate = (
            (efficiency < threshold)
            & z.between(-outer, -inner)
            & recovering
        )
        valid = indicators.notna().all(axis=1) & (indicators['atr'] > 0.0)
        trend_candidate = (trend_candidate & valid).fillna(False)
        congestion_candidate = (congestion_candidate & valid).fillna(False)
        confirmed = (
            (trend_candidate & trend_candidate.shift(1, fill_value=False))
            | (congestion_candidate & congestion_candidate.shift(1, fill_value=False))
        ).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_fill = False
        entry_price = np.nan
        entry_atr = np.nan
        fill_bar = -1
        hold_bars = max(1, int(params.holding_bars))
        stop_multiple = max(0.0, float(params.stop_atr))
        for i in range(len(data)):
            if pending_fill:
                # This open executes the preceding close's entry decision.
                entry_price = opens[i]
                fill_bar = i
                pending_fill = False
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    active = False
            if active:
                # Entry ATR and entry price remain fixed throughout the trade.
                stop_price = entry_price - stop_multiple * entry_atr
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - fill_bar >= hold_bars - 1
                if stopped or expired:
                    active = False
                else:
                    signals[i] = 1
                # An exit decision cannot also initiate a new trade.
                continue
            if confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                active = True
                pending_fill = True
                entry_atr = atr[i]
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
