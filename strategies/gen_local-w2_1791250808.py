from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalGapParams:
    frac_order: float = 0.55
    frac_window: int = 32
    reference_window: int = 20
    atr_window: int = 14
    gap_atr: float = 0.20
    stop_atr: float = 1.50
    hold_days: int = 2

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError('frac_order must be between zero and one')
        if min(self.frac_window, self.reference_window, self.atr_window) < 2:
            raise ValueError('lookbacks must be at least two')
        if not np.isfinite(self.gap_atr) or self.gap_atr <= 0:
            raise ValueError('gap_atr must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if self.hold_days not in (1, 2):
            raise ValueError('hold_days must be one or two')


class GeneratedStrategy(BaseStrategy[FractionalGapParams]):
    strategy_id = 'gen_local-w2_1791250808'

    @classmethod
    def params_type(cls):
        return FractionalGapParams

    def warmup_bars(self, params: FractionalGapParams) -> int:
        return int(max(200, params.frac_window + params.reference_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalGapParams) -> pd.DataFrame:
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        opening = data['open'].astype(float).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0))
        log_open = np.log(opening.where(opening > 0))
        fractional = log_close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * log_close.shift(lag)
        fractional_open = fractional - log_close + log_open
        equilibrium = fractional.shift(1).rolling(
            params.reference_window, min_periods=params.reference_window
        ).mean()
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        gap = opening - previous_close
        prior_atr = atr.shift(1)
        ma200 = close.rolling(200, min_periods=200).mean()
        previous_fractional = fractional.shift(1)
        out = pd.DataFrame({
            'fractional_close': fractional,
            'fractional_open': fractional_open,
            'previous_fractional_close': previous_fractional,
            'equilibrium': equilibrium,
            'true_range': true_range,
            'atr': atr,
            'prior_atr': prior_atr,
            'gap': gap,
            'ma200': ma200,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (atr > 0) & (prior_atr > 0)
        long_setup = (
            ready & (close > ma200)
            & (previous_fractional > equilibrium)
            & (fractional_open < equilibrium)
            & (fractional > equilibrium)
            & (gap < -params.gap_atr * prior_atr)
        )
        short_setup = (
            ready & (close < ma200)
            & (previous_fractional < equilibrium)
            & (fractional_open > equilibrium)
            & (fractional < equilibrium)
            & (gap > params.gap_atr * prior_atr)
        )
        out['setup'] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FractionalGapParams) -> SignalFrame:
        opening = data['open'].to_numpy(dtype=float)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        setup = indicators['setup'].to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        entry_bar = 0
        distance = 0.0
        for i in range(len(data)):
            if position:
                # The previous decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opening[i]
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(close[i])
                # Entry price and decision-bar ATR remain fixed for this trade.
                stopped = valid and position * (close[i] - entry) < -distance
                expired = i - entry_bar + 1 >= params.hold_days
                if not valid or stopped or expired:
                    position = 0
                    entry = np.nan
                raw[i] = position
                continue
            if setup[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(setup[i])
                entry = np.nan
                entry_bar = i + 1
                distance = params.stop_atr * atr[i]
                raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = FractionalGapParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0,
        'high': 103.0,
        'low': 97.0,
        'close': [100.0, 98.0, 100.0, 100.0, 102.0, 100.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'setup': [1, 0, 0, -1, 0, 0, 0],
        'atr': [1.0, 100.0, 1.0, 1.0, 100.0, 1.0, 1.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
    data['close'] = 101.0
    indicators['setup'] = [1, 0, 0, 0, 0, 0, 0]
    indicators['atr'] = 1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
