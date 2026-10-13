from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeComplexityParams:
    compression_window: int = 6
    baseline_window: int = 30
    ordinal_window: int = 12
    compression_ratio: float = 0.78
    complexity_min: float = 0.10
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.9
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeComplexityParams]):
    strategy_id = 'gen_local_1790988619'

    @classmethod
    def params_type(cls) -> type[RangeComplexityParams]:
        return RangeComplexityParams

    def warmup_bars(self, params: RangeComplexityParams) -> int:
        return int(max(
            params.baseline_window + 1,
            params.ordinal_window + 3,
            params.compression_window + 1,
            params.atr_window + 1,
        ) + 2)

    def indicators(self, data: pd.DataFrame, params: RangeComplexityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['short_range'] = out['true_range'].rolling(params.compression_window).mean()
        out['baseline_range'] = out['true_range'].rolling(params.baseline_window).mean()
        out['compression'] = out['short_range'].div(out['baseline_range'].replace(0.0, np.nan))
        out['box_high'] = data['high'].rolling(params.compression_window).max()
        out['box_low'] = data['low'].rolling(params.compression_window).min()

        # Stable temporal ordering resolves equal ranges without random jitter.
        a = out['true_range'].shift(2)
        b = out['true_range'].shift(1)
        c = out['true_range']
        valid = a.notna() & b.notna() & c.notna()
        out['ordinal_code'] = (
            4 * (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + (b > c).astype(int)
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        disequilibrium = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            name = 'ordinal_probability_' + str(code)
            out[name] = out['ordinal_code'].eq(code).astype(float).where(valid).rolling(params.ordinal_window).mean()
            probability = out[name]
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
            disequilibrium = disequilibrium + (probability - 1.0 / 6.0).pow(2)
        out['complexity'] = (entropy / np.log(6.0)) * (disequilibrium / (5.0 / 6.0))
        out['eligible'] = (
            out['compression'].le(params.compression_ratio)
            & out['complexity'].ge(params.complexity_min)
            & out['atr'].gt(0.0)
        )
        out['first_up'] = out['eligible'].shift(1, fill_value=False) & data['close'].gt(out['box_high'].shift(1))
        out['first_down'] = out['eligible'].shift(1, fill_value=False) & data['close'].lt(out['box_low'].shift(1))
        out['confirmed_up'] = (
            out['first_up'].shift(1, fill_value=False)
            & data['close'].gt(out['box_high'].shift(2))
            & data['close'].gt(previous_close)
        )
        out['confirmed_down'] = (
            out['first_down'].shift(1, fill_value=False)
            & data['close'].lt(out['box_low'].shift(2))
            & data['close'].lt(previous_close)
        )
        out['entry_direction'] = np.select(
            [out['confirmed_up'], out['confirmed_down']], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeComplexityParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry_direction'].to_numpy(dtype=int)
        side = 0
        pending_entry = 0
        entry = 0.0
        stop = 0.0
        armed = False
        age = 0
        hold_limit = min(2, max(1, int(params.max_hold_bars)))
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending_entry:
                # A close decision from yesterday fills at today's open.
                if np.isfinite(opens[i]) and opens[i] > 0.0 and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    side = pending_entry
                    entry = opens[i]
                    # Signed-price coordinates make both protective stops
                    # ratchet upward: long stops rise, short buy-stops fall.
                    stop = side * entry - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                pending_entry = 0

            if side:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    side = 0
                    continue
                signed_close = side * closes[i]
                signed_entry = side * entry
                exit_now = signed_close <= stop
                if signed_close - signed_entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
                    exit_now = exit_now or signed_close <= stop
                if exit_now or age >= hold_limit:
                    # Closing stop decisions execute on the following open.
                    side = 0
                    continue
                decisions[i] = side
            elif i >= warmup and candidates[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                pending_entry = int(candidates[i])
                decisions[i] = pending_entry

        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
