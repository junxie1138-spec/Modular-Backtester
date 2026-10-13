from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    envelope_window: int = 20
    compression_window: int = 3
    baseline_window: int = 12
    pressure_window: int = 3
    atr_window: int = 14
    edge_fraction: float = 0.20
    compression_ratio: float = 0.80
    pressure_threshold: float = 0.15
    stop_atr: float = 1.50
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791293900"

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(params.envelope_window, params.compression_window,
                       params.baseline_window, params.pressure_window,
                       params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        p = params
        result = pd.DataFrame(index=data.index)
        daily_range = (data['high'] - data['low']).clip(lower=0.0)
        envelope_high = data['high'].rolling(p.envelope_window).max()
        envelope_low = data['low'].rolling(p.envelope_window).min()
        envelope_width = (envelope_high - envelope_low).replace(0.0, np.nan)
        recent_range = daily_range.rolling(p.compression_window).mean()
        baseline_range = daily_range.rolling(p.baseline_window).mean()
        candle_body = (data['close'] - data['open']) / daily_range.replace(0.0, np.nan)
        candle_body = candle_body.clip(-1.0, 1.0).fillna(0.0)
        volume = data['volume'].clip(lower=0.0)
        signed_volume = candle_body * volume
        pressure_volume = signed_volume.rolling(p.pressure_window).sum()
        total_volume = volume.rolling(p.pressure_window).sum().replace(0.0, np.nan)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            daily_range,
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        result['daily_range'] = daily_range
        result['envelope_high'] = envelope_high
        result['envelope_low'] = envelope_low
        result['envelope_width'] = envelope_width
        result['location'] = (data['close'] - envelope_low) / envelope_width
        result['recent_range'] = recent_range
        result['baseline_range'] = baseline_range
        result['compression'] = recent_range / baseline_range.replace(0.0, np.nan)
        result['candle_body'] = candle_body
        result['signed_volume'] = signed_volume
        result['pressure_volume'] = pressure_volume
        result['total_volume'] = total_volume
        result['pressure'] = pressure_volume / total_volume
        result['previous_close'] = previous_close
        result['true_range'] = true_range
        result['atr'] = true_range.rolling(p.atr_window).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        p = params
        if not (0.0 < p.edge_fraction < 0.5 and p.compression_ratio > 0.0
                and 0.0 < p.pressure_threshold <= 1.0 and p.stop_atr > 0.0
                and 3 <= p.hold_bars <= 5):
            raise ValueError('Invalid entry, stop, or holding parameters')
        ready = indicators[['location', 'compression', 'pressure', 'atr']].notna().all(axis=1)
        ready &= indicators['atr'].gt(0.0)
        ready.iloc[:self.warmup_bars(p)] = False
        compressed = indicators['compression'].le(p.compression_ratio)
        long_entry = (ready & compressed
                      & indicators['location'].le(p.edge_fraction)
                      & indicators['pressure'].ge(p.pressure_threshold))
        short_entry = (ready & compressed
                       & indicators['location'].ge(1.0 - p.edge_fraction)
                       & indicators['pressure'].le(-p.pressure_threshold))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        pending_fill = False
        entry_price = np.nan
        fixed_distance = np.nan
        held = 0
        for i in range(len(data)):
            if pending_fill:
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    raise ValueError('Entry requires a finite positive next-bar open')
                entry_price = opens[i]
                pending_fill = False
            if position:
                held += 1
                stop_hit = (np.isfinite(closes[i])
                            and position * (closes[i] - entry_price) <= -fixed_distance)
                if stop_hit or held >= p.hold_bars:
                    position = 0
                    held = 0
            elif entries[i] and i + 1 < len(data):
                position = int(entries[i])
                fixed_distance = p.stop_atr * atrs[i]
                pending_fill = True
                held = 0
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = targets
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        assert df['signal'].isin([-1, 0, 1]).all() and df['size'].gt(0.0).all()
        return SignalFrame(data=df, signal_column='signal', size_column='size')
