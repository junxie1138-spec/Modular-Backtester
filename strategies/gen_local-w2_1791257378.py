from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    fractional_order: float = 0.4
    fractional_window: int = 48
    memory_window: int = 16
    gap_window: int = 10
    peak_window: int = 63
    min_drawdown: float = 0.03
    min_gap_debt: float = 0.004
    atr_window: int = 14
    initial_stop_atr: float = 3.0
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791257378'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.fractional_window + params.memory_window,
            params.gap_window + 1,
            params.peak_window,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        previous = close.shift(1).replace(0.0, np.nan)
        opening = data['open'].astype(float).replace(0.0, np.nan)
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        fractional = fractional / close.replace(0.0, np.nan)
        memory = fractional.rolling(params.memory_window, min_periods=params.memory_window).mean()
        overnight = opening / previous - 1.0
        intraday = close / opening - 1.0
        gap_debt = (-overnight.clip(upper=0.0)).rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()
        session_payment = intraday.rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = ranges.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return pd.DataFrame({
            'fractional_close': fractional,
            'fractional_memory': memory,
            'gap_debt': gap_debt,
            'session_payment': session_payment,
            'drawdown': close / peak.replace(0.0, np.nan) - 1.0,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        fractional = indicators['fractional_close']
        memory = indicators['fractional_memory']
        memory_recovery = (
            (fractional > memory)
            & (fractional.shift(1) <= memory.shift(1))
        )
        gap_repayment = (
            (indicators['gap_debt'] >= params.min_gap_debt)
            & (indicators['session_payment'] > indicators['gap_debt'])
        )
        entry_condition = (
            memory_recovery
            & gap_repayment
            & (indicators['drawdown'] <= -params.min_drawdown)
            & (indicators['atr'] > 0.0)
            & indicators.notna().all(axis=1)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        highest_close = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                # The prior close's decision fills at this bar's open.
                entry_price = opens[i]
                stop = entry_price - params.initial_stop_atr * atrs[i - 1]
                highest_close = entry_price
                age = 0
                armed = False
                active = True
                pending = False
            if active:
                age += 1
                if np.isfinite(closes[i]) and np.isfinite(atrs[i]):
                    highest_close = max(highest_close, closes[i])
                    if closes[i] >= entry_price * (1.0 + params.breakeven_gain):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price, highest_close - params.trail_atr * atrs[i])
                    # Close-confirmed exits execute at the following open.
                    if closes[i] <= stop or age >= params.max_hold_bars:
                        active = False
                elif age >= params.max_hold_bars:
                    active = False
                signals[i] = int(active)
            elif entry_condition[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
