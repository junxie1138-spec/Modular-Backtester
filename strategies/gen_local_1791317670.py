from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 30
    atr_window: int = 14
    min_autocorr: float = 0.20
    quiet_threshold: float = 0.25
    trail_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791317670'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.lookback + 3, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['range_fraction'] = out['true_range'].div(
            out['previous_close'].where(out['previous_close'] > 0)
        )
        out['lagged_range'] = out['range_fraction'].shift(1)
        window = out['range_fraction'].rolling(
            params.lookback, min_periods=params.lookback
        )
        out['range_mean'] = window.mean().shift(1)
        out['range_std'] = window.std(ddof=0).shift(1)
        out['range_autocorr'] = window.corr(
            out['lagged_range']
        ).shift(1).clip(-1.0, 1.0)
        out['range_forecast'] = (
            out['range_mean'] + out['range_autocorr']
            * (out['lagged_range'] - out['range_mean'])
        ).clip(lower=0.0)
        out['forecast_error'] = (
            out['range_fraction'] - out['range_forecast']
        ).div(out['range_std'].where(out['range_std'] > 0))
        out['close_return'] = data['close'].div(
            out['previous_close'].where(out['previous_close'] > 0)
        ) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out['candidate'] = (
            (out['range_autocorr'] >= params.min_autocorr)
            & (out['forecast_error'] <= -params.quiet_threshold)
            & (out['close_return'] > 0)
            & (data['close'] > data['open'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        candidate = indicators['candidate'].fillna(False).astype(bool)
        confirmed = candidate & candidate.shift(1, fill_value=False)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        held_sessions = 0
        highest_close = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if active:
                # Entry was filled at this session's open after the prior signal.
                held_sessions += 1
                if valid:
                    highest_close = max(highest_close, close)
                    stop = max(stop, highest_close - params.trail_atr * atr)
                if not valid or close <= stop or held_sessions >= 2:
                    active = False
                else:
                    signal[i] = 1
            elif valid and entries[i]:
                active = True
                held_sessions = 0
                # Only closes following the actual entry fill enter the trail.
                highest_close = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [100, 101, 103, 104, 105, 106, 107]}, index=index)
    indicators = pd.DataFrame({
        'candidate': [True, True, False, False, False, False, False],
        'atr': [1.0] * 7,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert (result['size'] > 0).all()
    indicators['candidate'] = [True, False, True, False, False, False, False]
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert not result['signal'].any()


if __name__ == '__main__':
    _self_check()
