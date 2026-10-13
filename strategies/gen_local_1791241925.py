from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    correlation_window: int = 40
    correlation_threshold: float = 0.10
    atr_period: int = 14
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791241925"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(max(3, params.correlation_window) + 2, max(2, params.atr_period) + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = max(3, int(params.correlation_window))
        atr_period = max(2, int(params.atr_period))
        previous_close = data['close'].shift(1).where(lambda s: s > 0)
        opening = data['open'].where(lambda s: s > 0)
        gap = (opening / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        intraday = (data['close'] / opening - 1).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out['gap'] = gap
        out['intraday'] = intraday
        out['gap_mean'] = gap.rolling(window, min_periods=window).mean()
        out['intraday_mean'] = intraday.rolling(window, min_periods=window).mean()
        out['gap_correlation'] = gap.rolling(window, min_periods=window).corr(gap.shift(1))
        out['intraday_correlation'] = intraday.rolling(window, min_periods=window).corr(intraday.shift(1))
        out['gap_forecast'] = out['gap_mean'] + out['gap_correlation'] * (gap - out['gap_mean'])
        out['intraday_forecast'] = out['intraday_mean'] + out['intraday_correlation'] * (intraday - out['intraday_mean'])
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_period, min_periods=atr_period).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        threshold = max(0.0, float(params.correlation_threshold))
        out['gap_agrees'] = (
            (gap < 0)
            & (out['gap_correlation'] < -threshold)
            & (out['gap_forecast'] > 0)
        ).fillna(False)
        out['intraday_agrees'] = (
            (intraday > 0)
            & (out['intraday_correlation'] > threshold)
            & (out['intraday_forecast'] > 0)
        ).fillna(False)
        out['entry'] = (
            out['gap_agrees'] & out['intraday_agrees'] & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        multiplier = max(0.01, float(params.trail_atr))
        for i in range(len(data)):
            if active:
                age += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                if valid_close:
                    high_water = closes[i] if not np.isfinite(high_water) else max(high_water, closes[i])
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, high_water - multiplier * atr[i])
                if age >= 4 or (valid_close and closes[i] <= stop):
                    active = False
                else:
                    signal[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [100., 102., 100., 100., 100., 100., 100.]}, index=index)
    indicators = pd.DataFrame({'atr': 1., 'entry': [True, False, False, False, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data['close'] = 100.
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
