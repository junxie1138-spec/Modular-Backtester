from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.4
    frac_window: int = 40
    correlation_window: int = 60
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8
    recovery_level: float = -0.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791229865'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.frac_window + params.correlation_window + 5, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError('frac_order must lie between zero and one')
        if min(params.frac_window, params.correlation_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two')
        if params.stop_atr <= 0.0 or params.holding_bars < 1:
            raise ValueError('Stop multiplier and holding period must be positive')
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['previous_close'] = close.shift(1)
        ranges = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1)
        out['true_range'] = ranges.max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(lag)
        out['fractional_close'] = fractional
        out['fractional_lag1'] = fractional.shift(1)
        out['fractional_lag5'] = fractional.shift(5)
        window = fractional.rolling(params.correlation_window, min_periods=params.correlation_window)
        out['fractional_mean'] = window.mean()
        out['fractional_std'] = window.std(ddof=0)
        out['phase_z'] = (fractional - out['fractional_mean']) / out['fractional_std'].replace(0.0, np.nan)
        out['phase_z_previous'] = out['phase_z'].shift(1)
        out['daily_correlation'] = window.corr(out['fractional_lag1'])
        out['weekly_correlation'] = window.corr(out['fractional_lag5'])
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            (close > out['ma200'])
            & (out['daily_correlation'] > 0.15)
            & (out['weekly_correlation'] < -0.2)
            & (out['phase_z_previous'] <= params.recovery_level)
            & (out['phase_z'] > params.recovery_level)
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        regime = indicators['ma200'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending_fill = False
        stop = np.nan
        entry_atr = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                if pending_fill:
                    # Entry executes at this bar's open, following yesterday's decision.
                    # Freeze the stop using that open and the decision bar's ATR.
                    stop = opening[i] - params.stop_atr * entry_atr
                    pending_fill = False
                held += 1
                if (
                    not np.isfinite(close[i])
                    or not np.isfinite(stop)
                    or not np.isfinite(regime[i])
                    or close[i] < stop
                    or close[i] <= regime[i]
                    or held >= params.holding_bars
                ):
                    active = False
                    signal[i] = 0
                    continue
            elif (
                entries[i]
                and np.isfinite(close[i])
                and np.isfinite(atr[i])
                and atr[i] > 0.0
                and np.isfinite(regime[i])
                and close[i] > regime[i]
            ):
                active = True
                pending_fill = True
                entry_atr = atr[i]
                held = 0
            signal[i] = int(active)
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': [100.0, 101.0, 110.0, 98.0, 96.0],
        'high': [101.0, 111.0, 111.0, 99.0, 100.0],
        'low': [99.0, 100.0, 97.0, 95.0, 95.0],
        'close': [100.0, 110.0, 98.0, 96.0, 99.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [2.0, 8.0, 1.0, 1.0, 1.0],
        'ma200': [90.0] * 5,
        'entry': [True, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params())
    # Entry at 101 fixes the stop at 97 despite subsequent price and ATR changes.
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0]
    assert result.data['size'].gt(0.0).all()
    timed = GeneratedStrategy.generate_signals(None, data, indicators, None, Params(holding_bars=1))
    assert timed.data['signal'].tolist() == [0, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
