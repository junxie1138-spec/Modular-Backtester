from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAbsorptionParams:
    breakout_bars: int = 20
    absorption_bars: int = 10
    atr_bars: int = 14
    min_gap_atr: float = 0.35
    min_supply_atr: float = 1.0
    min_absorption: float = 0.8
    stop_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapAbsorptionParams]):
    strategy_id = 'gen_local_1791340820'

    @classmethod
    def params_type(cls) -> type[GapAbsorptionParams]:
        return GapAbsorptionParams

    def warmup_bars(self, params: GapAbsorptionParams) -> int:
        return int(max(params.breakout_bars, params.absorption_bars, params.atr_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: GapAbsorptionParams) -> pd.DataFrame:
        if min(params.breakout_bars, params.absorption_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError('Position sizes must satisfy 0 < min_size <= max_size <= 1.')
        if not all(np.isfinite(v) and v > 0.0 for v in (
            params.min_gap_atr, params.min_supply_atr, params.min_absorption, params.stop_atr
        )):
            raise ValueError('Signal thresholds and stop multiplier must be finite and positive.')

        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        out['gap'] = data['open'] - out['prior_close']
        out['session_change'] = data['close'] - data['open']
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['prior_close']).abs(),
            (data['low'] - out['prior_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['prior_atr'] = out['atr'].shift(1).where(lambda s: s > 0.0)
        out['ceiling'] = data['high'].rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max().shift(1)
        out['negative_gap'] = (-out['gap']).clip(lower=0.0)
        out['negative_gap_session'] = out['session_change'].where(
            out['gap'] < 0.0, 0.0
        ).where(out['gap'].notna())
        out['prior_supply'] = out['negative_gap'].rolling(
            params.absorption_bars, min_periods=params.absorption_bars
        ).sum().shift(1)
        out['prior_recovery'] = out['negative_gap_session'].rolling(
            params.absorption_bars, min_periods=params.absorption_bars
        ).sum().shift(1)
        out['absorption'] = out['prior_recovery'].div(
            out['prior_supply'].where(out['prior_supply'] > 0.0)
        )
        out['supply_atr'] = out['prior_supply'] / out['prior_atr']
        out['gap_atr'] = out['gap'] / out['prior_atr']
        out['strength'] = (
            0.5 * (out['gap_atr'] / (2.0 * params.min_gap_atr)).clip(0.0, 1.0)
            + 0.5 * (out['absorption'] / (2.0 * params.min_absorption)).clip(0.0, 1.0)
        ).fillna(0.0)
        out['entry'] = (
            (data['open'] > out['ceiling'])
            & (data['close'] >= data['open'])
            & (out['gap_atr'] >= params.min_gap_atr)
            & (out['supply_atr'] >= params.min_supply_atr)
            & (out['absorption'] >= params.min_absorption)
            & np.isfinite(out['prior_atr'])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAbsorptionParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=bool)
        strength = indicators['strength'].to_numpy(dtype=float)

        active = False
        pending_entry = False
        pending_atr = 0.0
        stop_price = 0.0
        entry_bar = 0
        position_size = params.min_size

        for i in range(count):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_bar = i
                    stop_price = opens[i] - params.stop_atr * pending_atr

            exited = False
            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    active = False
                    exited = True

            if not active and not exited and entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                pending_entry = True
                pending_atr = atr[i]
                position_size = float(
                    params.min_size
                    + (params.max_size - params.min_size) * np.clip(strength[i], 0.0, 1.0)
                )

            if active or pending_entry:
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
