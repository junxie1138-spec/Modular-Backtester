from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RelayParams:
    atr_period: int = 20
    gap_threshold: float = 0.35
    expansion_threshold: float = 1.40
    stop_atr: float = 2.50
    hold_bars: int = 18
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RelayParams]):
    strategy_id = 'gen_local-w2_1790996701'

    @classmethod
    def params_type(cls) -> type[RelayParams]:
        return RelayParams

    def warmup_bars(self, params: RelayParams) -> int:
        return max(1, int(params.atr_period)) + 3

    def indicators(self, data: pd.DataFrame, params: RelayParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['range'] = (data['high'] - data['low']).clip(lower=0.0)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        period = max(1, int(params.atr_period))
        out['atr'] = out['true_range'].rolling(period, min_periods=period).mean()
        out['gap'] = data['open'] - previous_close
        out['normalized_gap'] = out['gap'] / out['atr'].shift(1).replace(0.0, np.nan)
        out['seed_gap'] = out['normalized_gap'].shift(2)
        out['gap_midpoint'] = ((data['open'] + previous_close) / 2.0).shift(2)
        out['expansion'] = out['range'] / out['range'].shift(2).replace(0.0, np.nan)
        out['expanding_relay'] = (
            (out['range'].shift(1) > out['range'].shift(2))
            & (out['range'] > out['range'].shift(1))
        )
        out['up_translation'] = (
            (data['high'].shift(1) > data['high'].shift(2))
            & (data['low'].shift(1) > data['low'].shift(2))
            & (data['high'] > data['high'].shift(1))
            & (data['low'] > data['low'].shift(1))
            & (data['low'].shift(1) > out['gap_midpoint'])
            & (data['low'] > out['gap_midpoint'])
        )
        out['down_translation'] = (
            (data['high'].shift(1) < data['high'].shift(2))
            & (data['low'].shift(1) < data['low'].shift(2))
            & (data['high'] < data['high'].shift(1))
            & (data['low'] < data['low'].shift(1))
            & (data['high'].shift(1) < out['gap_midpoint'])
            & (data['high'] < out['gap_midpoint'])
        )
        gap_threshold = max(float(params.gap_threshold), 1e-9)
        expansion_threshold = max(float(params.expansion_threshold), 1e-9)
        out['strength'] = (
            out['seed_gap'].abs() / gap_threshold
            * out['expansion'] / expansion_threshold
        ).replace([np.inf, -np.inf], np.nan)
        ready = (
            out['atr'].notna() & (out['atr'] > 0.0)
            & out['seed_gap'].notna() & out['strength'].notna()
        )
        common = ready & out['expanding_relay'] & (out['expansion'] >= expansion_threshold)
        long_entry = common & (out['seed_gap'] >= gap_threshold) & out['up_translation']
        short_entry = common & (out['seed_gap'] <= -gap_threshold) & out['down_translation']
        out['entry_direction'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RelayParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        directions = indicators['entry_direction'].to_numpy(dtype=int)
        strengths = indicators['strength'].to_numpy(dtype=float)
        minimum_size = max(0.01, min(1.0, float(params.minimum_size)))
        maximum_size = max(minimum_size, min(1.0, float(params.maximum_size)))
        hold_bars = max(1, int(params.hold_bars))
        stop_multiple = max(0.01, float(params.stop_atr))
        position = 0
        pending_direction = 0
        pending_atr = 0.0
        position_size = minimum_size
        fixed_stop = np.nan
        age = 0

        for i in range(count):
            # A decision from the previous close fills at this bar's open.
            if pending_direction:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = pending_direction
                    fixed_stop = opens[i] - position * stop_multiple * pending_atr
                    age = 0
                pending_direction = 0

            if position:
                age += 1
                stopped = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < fixed_stop)
                    or (position == -1 and closes[i] > fixed_stop)
                )
                sizes[i] = position_size
                if stopped or age >= hold_bars:
                    position = 0
                    fixed_stop = np.nan
                    # The shifted zero exits at the following bar's open.
                    continue
                signals[i] = position
                continue

            if directions[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(strengths[i]):
                pending_direction = int(directions[i])
                pending_atr = float(atr[i])
                scaled_strength = float(np.clip((strengths[i] - 1.0) / 2.0, 0.0, 1.0))
                position_size = minimum_size + (maximum_size - minimum_size) * scaled_strength
                signals[i] = pending_direction
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
