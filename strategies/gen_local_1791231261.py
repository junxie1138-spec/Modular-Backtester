from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryParams:
    volume_window: int = 20
    volume_multiple: float = 1.05
    minimum_return: float = 0.001
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_gain: float = 0.015
    trail_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExpiryParams]):
    strategy_id = 'gen_local_1791231261'

    @classmethod
    def params_type(cls) -> type[ExpiryParams]:
        return ExpiryParams

    @staticmethod
    def warmup_bars(params: ExpiryParams) -> int:
        return int(max(params.volume_window, params.atr_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ExpiryParams) -> pd.DataFrame:
        if params.volume_window < 1 or params.atr_window < 1:
            raise ValueError('Indicator windows must be positive.')
        if params.holding_bars < 1:
            raise ValueError('Holding period must be positive.')
        values = (params.volume_multiple, params.minimum_return,
                  params.initial_stop_atr, params.breakeven_gain,
                  params.trail_atr)
        if not all(np.isfinite(value) for value in values):
            raise ValueError('Parameters must be finite.')
        if min(params.volume_multiple, params.initial_stop_atr,
               params.breakeven_gain, params.trail_atr) <= 0:
            raise ValueError('Volume and stop parameters must be positive.')
        if params.minimum_return < 0:
            raise ValueError('Minimum return must be nonnegative.')

        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        out['atr'] = true_range.rolling(params.atr_window).mean()
        out['return'] = data['close'].div(previous_close.where(previous_close > 0)) - 1.0
        out['prior_volume_mean'] = data['volume'].rolling(params.volume_window).mean().shift(1)
        out['relative_volume'] = data['volume'].div(
            out['prior_volume_mean'].where(out['prior_volume_mean'] > 0)
        ).replace([np.inf, -np.inf], np.nan)

        index = pd.DatetimeIndex(data.index)
        first_weekday = (index.dayofweek - (index.day - 1)) % 7
        third_friday = 15 + (4 - first_weekday) % 7
        out['days_after_expiry'] = index.day - third_friday
        out['expiry_window'] = out['days_after_expiry'].between(3, 10)
        out['qualified_move'] = (
            out['expiry_window']
            & out['return'].gt(params.minimum_return)
            & out['relative_volume'].ge(params.volume_multiple)
            & out['atr'].gt(0)
        ).fillna(False)
        out['confirmed_entry'] = (
            out['qualified_move']
            & out['qualified_move'].shift(1, fill_value=False)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ExpiryParams,
    ) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['confirmed_entry'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = np.nan
        stop = np.nan
        entry_atr = np.nan
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    active = False
                    continue
                stop = entry - params.initial_stop_atr * entry_atr
                held = 0

            if active:
                held += 1
                close = closes[i]
                low = lows[i]
                atr = atrs[i]
                if not np.isfinite(close) or not np.isfinite(low):
                    active = False
                    continue

                # Test the stop established before this bar's price action.
                if low <= stop or held >= params.holding_bars:
                    active = False
                    continue

                if close >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, close - params.trail_atr * atr)
                    if close <= stop:
                        active = False
                        continue
                decisions[i] = 1

            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                pending = True
                armed = False
                entry_atr = atrs[i]
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2024-01-01', periods=6, freq='B')
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [101.0, 101.0, 103.0, 105.0, 102.0, 101.0],
        'low': [99.5, 99.5, 100.5, 102.5, 100.5, 99.5],
        'close': [100.0, 100.0, 102.0, 104.0, 101.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({
        'atr': [1.0] * 6,
        'confirmed_entry': [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, ExpiryParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result.data['size'].gt(0).all()
    timed = GeneratedStrategy.generate_signals(
        data, derived, None, ExpiryParams(holding_bars=2)
    )
    assert timed.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
