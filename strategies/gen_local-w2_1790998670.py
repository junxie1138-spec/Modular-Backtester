from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    fractional_order: float = 0.45
    fractional_window: int = 40
    pressure_window: int = 24
    atr_window: int = 14
    minimum_gap_atr: float = 0.35
    stop_atr: float = 2.4
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = 'gen_local-w2_1790998670'

    @classmethod
    def params_type(cls):
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return int(max(params.fractional_window + params.pressure_window + 1,
                       params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError('fractional_order must be between zero and one')
        if min(params.fractional_window, params.pressure_window,
               params.atr_window, params.holding_bars) < 2:
            raise ValueError('window lengths and holding_bars must be at least two')
        if params.stop_atr <= 0.0 or params.minimum_gap_atr <= 0.0:
            raise ValueError('ATR multipliers must be positive')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(data[['high', 'low', 'close']].notna().all(axis=1))
        out['atr'] = true_range.rolling(params.atr_window,
                                        min_periods=params.atr_window).mean()

        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = weights[lag - 1] * (lag - 1 - params.fractional_order) / lag
        fractional_close = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        out['fractional_close'] = fractional_close
        out['fractional_baseline'] = fractional_close.rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).mean()
        out['fractional_change'] = fractional_close.diff()
        out['gap_atr'] = (data['open'] - previous_close) / out['atr'].shift(1).replace(0.0, np.nan)

        pressure = (
            out['fractional_close'].gt(out['fractional_baseline'])
            & out['fractional_change'].gt(0.0)
            & np.isfinite(out['fractional_close'])
            & np.isfinite(out['fractional_baseline'])
        )
        out['pressure_confirmed'] = pressure
        out['gap_seed'] = (
            out['gap_atr'].ge(params.minimum_gap_atr)
            & data['open'].gt(data['high'].shift(1))
            & close.ge(data['open'])
            & pressure
            & np.isfinite(out['gap_atr'])
        )
        out['entry_confirmed'] = (
            out['gap_seed'].shift(1, fill_value=False)
            & pressure
            & close.ge(data['open'].shift(1))
            & data['low'].gt(close.shift(2))
            & out['atr'].gt(0.0)
            & np.isfinite(out['atr'])
        )
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=np.int64)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_confirmed'].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        frozen_atr = 0.0
        stop_price = np.nan

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Read the execution bar's open only after that bar arrives.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        active = False
                        continue
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                stopped = not np.isfinite(closes[i]) or closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    active = False
                    continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                stop_price = np.nan
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')
