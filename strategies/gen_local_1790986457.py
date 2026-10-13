from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 20
    entropy_floor: float = 0.85
    channel_window: int = 20
    reclaim_level: float = 0.35
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790986457'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.entropy_window + 5,
                       params.channel_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Earlier observations rank below later observations on exact ties.
        pattern = (4 * (a > b).astype(int)
                   + 2 * (a > c).astype(int)
                   + (b > c).astype(int)).where(valid)
        out['ordinal_pattern'] = pattern
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = (pattern.eq(code).astype(float).where(valid)
                           .rolling(params.entropy_window,
                                    min_periods=params.entropy_window).mean())
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.where(probability > 0, 1.0))
        out['entropy'] = entropy / np.log(6.0)
        out['entropy_rise'] = out['entropy'].diff(3)
        out['channel_low'] = data['low'].rolling(
            params.channel_window, min_periods=params.channel_window).min().shift(1)
        out['channel_high'] = data['high'].rolling(
            params.channel_window, min_periods=params.channel_window).max().shift(1)
        width = out['channel_high'] - out['channel_low']
        out['relative_position'] = (close - out['channel_low']) / width.where(width > 0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['true_range'] = true_range
        out['atr'] = true_range.rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['complexity_agrees'] = (
            out['entropy'].ge(params.entropy_floor)
            & out['entropy_rise'].gt(0)
        ).fillna(False)
        out['position_agrees'] = (
            out['relative_position'].shift(1).lt(params.reclaim_level)
            & out['relative_position'].ge(params.reclaim_level)
            & out['relative_position'].lt(0.65)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entry_condition = (
            indicators['complexity_agrees'].fillna(False)
            & indicators['position_agrees'].fillna(False)
            & indicators['atr'].gt(0)
        ).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_price = np.nan
        stop = np.nan
        armed = False
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            # A decision from yesterday becomes exposure at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    entry_price = opens[i]
                    stop = entry_price - params.trail_atr * atr[i - 1]
                    armed = False
                    held = 0
                else:
                    decisions[i] = 0
                    continue

            if active:
                held += 1
                # Test the previously established stop before raising it.
                # Exit decisions execute at the following open after shifting.
                stop_hit = ((np.isfinite(opens[i]) and opens[i] <= stop)
                            or (np.isfinite(lows[i]) and lows[i] <= stop))
                if stop_hit or held >= params.max_hold_bars:
                    active = False
                    decisions[i] = 0
                    continue
                if np.isfinite(highs[i]):
                    if highs[i] >= entry_price * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, highs[i] - params.trail_atr * atr[i])
                decisions[i] = 1
            elif i >= warmup - 1 and entry_condition[i]:
                pending = True
                decisions[i] = 1

        df['signal'] = decisions
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
