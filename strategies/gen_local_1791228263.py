from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    extreme_window: int = 20
    volume_window: int = 20
    atr_window: int = 14
    extreme_fraction: float = 0.035
    shock_volume: float = 1.9
    shock_range_atr: float = 1.4
    quiet_volume: float = 1.0
    recovery_volume: float = 1.2
    anchor_lifetime: int = 8
    breakeven_gain: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791228263'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.extreme_window + 1,
                       params.volume_window + 3,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['range'] = data['high'] - data['low']
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['volume_baseline'] = data['volume'].shift(1).rolling(params.volume_window).mean()
        out['volume_ratio'] = data['volume'] / out['volume_baseline'].replace(0.0, np.nan)
        out['prior_peak'] = data['high'].shift(1).rolling(params.extreme_window).max()
        out['prior_trough'] = data['low'].shift(1).rolling(params.extreme_window).min()
        out['drawdown'] = 1.0 - close / out['prior_peak'].replace(0.0, np.nan)
        out['drawup'] = close / out['prior_trough'].replace(0.0, np.nan) - 1.0
        out['range_atr'] = out['range'] / out['atr'].replace(0.0, np.nan)
        shock = (out['volume_ratio'] >= params.shock_volume) & (out['range_atr'] >= params.shock_range_atr)
        out['down_shock'] = (shock & (close < data['open']) &
                             (out['drawdown'] >= params.extreme_fraction) &
                             (data['low'] <= out['prior_trough'])).fillna(False)
        out['up_shock'] = (shock & (close > data['open']) &
                           (out['drawup'] >= params.extreme_fraction) &
                           (data['high'] >= out['prior_peak'])).fillna(False)
        out['body_midpoint'] = (data['open'] + close) / 2.0
        out['bar_number'] = np.arange(len(data), dtype=float)
        for label, event in [('down', out['down_shock']), ('up', out['up_shock'])]:
            out[label + '_midpoint'] = out['body_midpoint'].where(event).ffill()
            out[label + '_low'] = data['low'].where(event).ffill()
            out[label + '_high'] = data['high'].where(event).ffill()
            out[label + '_anchor_bar'] = out['bar_number'].where(event).ffill()
            out[label + '_age'] = out['bar_number'] - out[label + '_anchor_bar']
        out['quiet_pair'] = (out['volume_ratio'].shift(1).rolling(2).max() < params.quiet_volume).fillna(False)
        renewed = out['volume_ratio'] >= params.recovery_volume
        out['long_entry'] = (
            renewed & out['quiet_pair'] &
            out['down_age'].between(3, params.anchor_lifetime) &
            (close > out['down_midpoint']) &
            (out['previous_close'] <= out['down_midpoint']) &
            (data['low'] > out['down_low'])
        ).fillna(False)
        out['short_entry'] = (
            renewed & out['quiet_pair'] &
            out['up_age'].between(3, params.anchor_lifetime) &
            (close < out['up_midpoint']) &
            (out['previous_close'] >= out['up_midpoint']) &
            (data['high'] < out['up_high'])
        ).fillna(False)
        out['long_strength'] = ((close - out['down_midpoint']) / out['atr'].replace(0.0, np.nan)) * out['volume_ratio']
        out['short_strength'] = ((out['up_midpoint'] - close) / out['atr'].replace(0.0, np.nan)) * out['volume_ratio']
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        if params.min_size <= 0 or params.max_size < params.min_size:
            raise ValueError('Position size bounds must be positive and ordered')
        if params.trail_atr <= 0 or params.breakeven_gain <= 0 or params.max_hold < 1:
            raise ValueError('Exit parameters must be positive')
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        long_entries = indicators['long_entry'].to_numpy(dtype=bool)
        short_entries = indicators['short_entry'].to_numpy(dtype=bool)
        long_strength = indicators['long_strength'].to_numpy(dtype=float)
        short_strength = indicators['short_strength'].to_numpy(dtype=float)
        down_anchors = indicators['down_anchor_bar'].to_numpy(dtype=float)
        up_anchors = indicators['up_anchor_bar'].to_numpy(dtype=float)
        side = 0
        pending = False
        entry = 0.0
        stop_progress = 0.0
        armed = False
        held = 0
        trade_size = params.min_size
        used_down = -1.0
        used_up = -1.0
        for i in range(n):
            if pending:
                pending = False
                entry = opens[i]
                prior_atr = atrs[i - 1]
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(prior_atr) or prior_atr <= 0:
                    side = 0
                    continue
                # Signed price progress lets both directions use an upward-only stop ratchet.
                stop_progress = -params.trail_atr * prior_atr
                armed = False
                held = 0
            if side:
                held += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    signals[i] = side
                    sizes[i] = trade_size
                    continue
                progress = side * (closes[i] - entry)
                if progress <= stop_progress or held >= params.max_hold:
                    side = 0
                    continue
                if progress >= params.breakeven_gain * entry:
                    armed = True
                if armed:
                    stop_progress = max(stop_progress, 0.0)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop_progress = max(stop_progress, progress - params.trail_atr * atrs[i])
                signals[i] = side
                sizes[i] = trade_size
                continue
            if i < GeneratedStrategy.warmup_bars(params) - 1:
                continue
            if not np.isfinite(atrs[i]) or atrs[i] <= 0:
                continue
            strength = np.nan
            if long_entries[i] and down_anchors[i] != used_down:
                side = 1
                strength = long_strength[i]
                used_down = down_anchors[i]
            elif short_entries[i] and up_anchors[i] != used_up:
                side = -1
                strength = short_strength[i]
                used_up = up_anchors[i]
            if side:
                if not np.isfinite(strength) or strength <= 0:
                    side = 0
                    continue
                confidence = strength / (1.0 + strength)
                trade_size = float(params.min_size + (params.max_size - params.min_size) * confidence)
                signals[i] = side
                sizes[i] = trade_size
                pending = True
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        assert df['signal'].isin([-1, 0, 1]).all()
        assert np.isfinite(df['size']).all() and (df['size'] > 0).all()
        return SignalFrame(data=df, signal_column='signal', size_column='size')
