from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RejectionParams:
    peak_window: int = 20
    percentile_window: int = 126
    tail_percentile: float = 0.85
    atr_window: int = 14
    activation_pct: float = 0.005
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0

    def __post_init__(self):
        if min(self.peak_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < self.tail_percentile < 1.0:
            raise ValueError('tail_percentile must be between zero and one.')
        if not all(np.isfinite(v) and v > 0.0 for v in (
            self.activation_pct, self.trail_atr, self.initial_stop_atr
        )):
            raise ValueError('Stop parameters must be finite and positive.')


class GeneratedStrategy(BaseStrategy[RejectionParams]):
    strategy_id = 'gen_local-w2_1791264942'

    @classmethod
    def params_type(cls):
        return RejectionParams

    @staticmethod
    def warmup_bars(params):
        return int(max(
            params.peak_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['peak'] = close.rolling(params.peak_window).max().shift(1)
        peak = out['peak'].where(out['peak'] > 0.0)
        out['depth'] = (1.0 - close / peak).clip(lower=0.0)
        out['low_depth'] = (1.0 - data['low'] / peak).clip(lower=0.0)
        out['boundary'] = out['depth'].rolling(
            params.percentile_window
        ).quantile(params.tail_percentile).shift(1)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['previous_depth'] = out['depth'].shift(1)
        out['entry'] = (
            (out['boundary'] > 0.0)
            & (out['previous_depth'] < out['boundary'])
            & (out['low_depth'] > out['boundary'])
            & (out['depth'] < out['boundary'])
            & (close > data['open'])
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        count = len(data)
        raw = np.zeros(count, dtype=int)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        high_water = 0.0
        age = 0

        for i in range(count):
            opening, high, low, close = prices[i]
            valid_prices = np.isfinite(prices[i]).all() and (prices[i] > 0.0).all()
            exited = False

            # A target decided yesterday first becomes active at today's open.
            if not active and i > 0 and raw[i - 1] == 1:
                if valid_prices and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    # ponytail: OHLC entry proxy; actual fill reconciliation needs execution feedback.
                    entry_price = opening
                    stop = entry_price - params.initial_stop_atr * atr[i - 1]
                    high_water = entry_price
                    age = 0
                    armed = False
                    active = True
                else:
                    exited = True

            if active:
                age += 1
                if not valid_prices or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    active = False
                    exited = True
                elif low <= stop:
                    # Stop breaches become next-open exits through the signal shift.
                    active = False
                    exited = True
                else:
                    high_water = max(high_water, high)
                    if high_water >= entry_price * (1.0 + params.activation_pct):
                        armed = True
                    if armed:
                        # Update at the close; never assume an intrabar high/low ordering.
                        stop = max(stop, entry_price, high_water - params.trail_atr * atr[i])
                    if close <= stop or age >= 2:
                        active = False
                        exited = True

            if active:
                raw[i] = 1
            elif not exited and valid_prices and entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                raw[i] = 1

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 100.1, 'low': 99.9,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'atr': 1.0, 'entry': [True, False, False, False, False],
    }, index=index)
    params = RejectionParams()
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], ['high', 'close']] = [104.0, 101.0]
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    assert (result['size'] > 0.0).all()
    prefix = GeneratedStrategy.generate_signals(data.iloc[:3], derived.iloc[:3], None, params).data
    assert result.iloc[:3].equals(prefix)


if __name__ == '__main__':
    _self_check()
