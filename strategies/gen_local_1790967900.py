from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FracTrendParams:
    frac_order: float = 0.45
    frac_window: int = 40
    baseline_window: int = 20
    efficiency_window: int = 5
    strength_threshold: float = 0.6
    efficiency_threshold: float = 0.3
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        windows = (self.frac_window, self.baseline_window,
                   self.efficiency_window, self.atr_window, self.hold_bars)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in windows):
            raise ValueError('Windows and holding period must be positive integers')
        if self.frac_window < 2 or not 0.0 < self.frac_order < 1.0:
            raise ValueError('Fractional order must lie between zero and one')
        if not np.isfinite(self.strength_threshold) or self.strength_threshold <= 0:
            raise ValueError('Strength threshold must be positive and finite')
        if not 0.0 < self.efficiency_threshold < 1.0:
            raise ValueError('Efficiency threshold must lie between zero and one')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('Stop multiplier must be positive and finite')


class GeneratedStrategy(BaseStrategy[FracTrendParams]):
    strategy_id = 'gen_local_1790967900'

    @classmethod
    def params_type(cls):
        return FracTrendParams

    @staticmethod
    def warmup_bars(params: FracTrendParams) -> int:
        return int(max(200,
                       params.frac_window + params.baseline_window,
                       params.frac_window + params.efficiency_window,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FracTrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            high - low,
            (high - out['previous_close']).abs(),
            (low - out['previous_close']).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['ma200'] = close.rolling(200).mean()

        # Fixed-width causal fractional difference: w[k] = w[k-1]*(k-1-d)/k.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        out['fractional_baseline'] = fractional.rolling(params.baseline_window).mean()
        out['fractional_step'] = fractional.diff()
        out['fractional_displacement'] = out['fractional_step'].rolling(params.efficiency_window).sum()
        out['fractional_path'] = out['fractional_step'].abs().rolling(params.efficiency_window).sum()
        out['efficiency'] = out['fractional_displacement'] / out['fractional_path'].where(out['fractional_path'] > 0)
        out['strength'] = (fractional - out['fractional_baseline']) / out['atr'].where(out['atr'] > 0)
        out['previous_strength'] = out['strength'].shift(1)
        out['entry'] = (
            (close > out['ma200'])
            & (out['strength'] > params.strength_threshold)
            & (out['previous_strength'] <= params.strength_threshold)
            & (out['efficiency'] > params.efficiency_threshold)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FracTrendParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        stop_price = np.nan

        for i in range(len(data)):
            if active:
                # The previous close's decision enters at this bar's open.
                # Freeze the signal-bar ATR and next-open entry reference.
                if i == entry_bar:
                    stop_price = opening[i] - params.stop_atr * entry_atr
                valid = (np.isfinite(close[i]) and np.isfinite(regime[i])
                         and np.isfinite(stop_price) and opening[entry_bar] > 0)
                exit_now = (
                    not valid
                    or close[i] < stop_price
                    or close[i] <= regime[i]
                    or i - entry_bar + 1 >= params.hold_bars
                )
                if exit_now:
                    active = False
                else:
                    decisions[i] = 1
            elif (entries[i] and np.isfinite(close[i]) and close[i] > 0
                  and np.isfinite(regime[i]) and close[i] > regime[i]
                  and np.isfinite(atr[i]) and atr[i] > 0):
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0] * 5,
        'high': [101.0, 105.0, 102.0, 101.0, 101.0],
        'low': [99.0, 99.0, 99.0, 96.0, 96.0],
        'close': [100.0, 104.0, 101.0, 97.0, 97.0],
        'volume': [1000.0] * 5
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 10.0, 10.0, 10.0, 10.0],
        'ma200': [90.0] * 5,
        'entry': [True, False, False, False, False]
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, FracTrendParams())
    # Fixed stop stays at 98 despite higher closes and changing ATR.
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0]
    assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
