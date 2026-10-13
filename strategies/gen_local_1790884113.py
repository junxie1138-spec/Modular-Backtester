from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeOverlapParams:
    regime_window: int = 8
    envelope_window: int = 20
    overlap_threshold: float = 0.55
    stretch_multiple: float = 1.25
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeOverlapParams]):
    strategy_id = 'gen_local_1790884113'

    @classmethod
    def params_type(cls) -> type[RangeOverlapParams]:
        return RangeOverlapParams

    @staticmethod
    def warmup_bars(params: RangeOverlapParams) -> int:
        return int(max(200, params.regime_window + 1, params.envelope_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeOverlapParams) -> pd.DataFrame:
        if params.regime_window < 2 or params.envelope_window < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.overlap_threshold < 1.0:
            raise ValueError('overlap_threshold must be between zero and one.')
        if params.stretch_multiple <= 0.0 or params.profit_target <= 0.0 or params.hold_bars < 1:
            raise ValueError('Stretch multiple, profit target, and holding period must be positive.')
        out = pd.DataFrame(index=data.index)
        high, low, close = data['high'], data['low'], data['close']
        out['range'] = (high - low).clip(lower=0.0)
        out['previous_high'] = high.shift(1)
        out['previous_low'] = low.shift(1)
        out['previous_range'] = out['range'].shift(1)
        out['intersection'] = (
            np.minimum(high, out['previous_high'])
            - np.maximum(low, out['previous_low'])
        ).clip(lower=0.0)
        out['overlap'] = out['intersection'] / np.minimum(
            out['range'], out['previous_range']
        ).replace(0.0, np.nan)
        out['overlap_regime'] = out['overlap'].rolling(params.regime_window).mean()
        out['baseline_range'] = out['range'].shift(1).rolling(params.envelope_window).median()
        out['envelope_low'] = low.shift(1).rolling(params.envelope_window).min()
        out['envelope_high'] = high.shift(1).rolling(params.envelope_window).max()
        out['close_location'] = (close - low) / out['range'].replace(0.0, np.nan)
        out['upper_step'] = high - out['previous_high']
        out['lower_step'] = low - out['previous_low']
        out['ma200'] = close.rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeOverlapParams,
    ) -> SignalFrame:
        ind = indicators
        ready = ind.notna().all(axis=1) & (ind['baseline_range'] > 0.0)
        bullish = data['close'] > ind['ma200']
        elastic = ind['overlap_regime'] >= params.overlap_threshold
        contained_recovery = (
            (ind['range'] >= params.stretch_multiple * ind['baseline_range'])
            & (data['low'] < ind['previous_low'])
            & (data['low'] >= ind['envelope_low'])
            & (data['high'] <= ind['envelope_high'])
            & (data['close'] > ind['previous_low'])
            & (ind['close_location'] >= 0.75)
        )
        upward_separation = (
            (ind['overlap'] <= 0.25)
            & (ind['upper_step'] > 0.0)
            & (ind['lower_step'] > 0.0)
            & (ind['range'] <= params.stretch_multiple * ind['baseline_range'])
            & (ind['close_location'] >= 0.65)
        )
        entries = (
            ready & bullish
            & ((elastic & contained_recovery) | (~elastic & upward_separation))
        ).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_up = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_up:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and i + 1 < len(data):
                active = True
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        held = 0
        for signal in df['signal'].to_numpy():
            held = held + 1 if signal == 1 else 0
            assert held <= params.hold_bars, 'Time stop exceeded its holding limit.'
        return SignalFrame(data=df, signal_column='signal', size_column='size')
