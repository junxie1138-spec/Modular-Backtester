from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    frac_order: float = 0.55
    frac_window: int = 32
    atr_window: int = 14
    min_gap_atr: float = 0.35
    min_release_atr: float = 0.05
    stop_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if not np.isfinite([self.frac_order, self.min_gap_atr, self.min_release_atr, self.stop_atr]).all():
            raise ValueError('Parameters must be finite')
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError('frac_order must lie between zero and one')
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.frac_window, self.atr_window, self.hold_bars)):
            raise ValueError('Window and holding parameters must be integers')
        if min(self.frac_window, self.atr_window) < 2 or not 3 <= self.hold_bars <= 5:
            raise ValueError('Invalid window or holding horizon')
        if self.min_gap_atr <= 0 or self.min_release_atr <= 0 or self.stop_atr <= 0:
            raise ValueError('ATR thresholds must be positive')


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = 'gen_local_1791275169'

    @classmethod
    def params_type(cls):
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(200, params.frac_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous = close.shift(1)
        out['previous_close'] = previous
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['previous_atr'] = out['atr'].shift(1)

        # Fixed-width fractional differencing, newest observation first.
        # ponytail: finite memory truncates the fractional filter; increase frac_window for a longer tail.
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fractional = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(k)
        out['fractional_close'] = fractional
        out['fractional_change'] = fractional.diff()
        out['close_change'] = close.diff()
        scale = out['previous_atr'].where(out['previous_atr'] > 0)
        out['gap_atr'] = (data['open'] - previous) / scale
        out['release_atr'] = out['fractional_change'] / scale
        valid = np.isfinite(out[['ma200', 'atr', 'previous_atr', 'gap_atr', 'release_atr', 'close_change']]).all(axis=1)
        valid &= np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        out['entry'] = (
            valid
            & (out['atr'] > 0)
            & (close > out['ma200'])
            & (out['gap_atr'] <= -params.min_gap_atr)
            & (out['close_change'] < 0)
            & (close > data['open'])
            & (out['release_atr'] >= params.min_release_atr)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapMemoryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_atr = np.nan
        stop = np.nan
        age = 0

        for i in range(len(data)):
            if active:
                if pending:
                    # This open executes yesterday's decision. Only today's
                    # close decision uses it; the final shift preserves causality.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        pending = False
                        continue
                    stop = opens[i] - params.stop_atr * entry_atr
                    pending = False
                age += 1
                # Freeze both entry price and signal-day ATR for the entire trade.
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                pending = True
                entry_atr = atr[i]
                age = 0
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'open': 100.0,
        'high': 112.0,
        'low': 96.0,
        'close': [110.0, 99.0, 97.0, 100.0, 110.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'entry': [True, False, False, False, True, False, False, False, False, False],
        'atr': [1.0] + [10.0] * 9,
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, GapMemoryParams()).data
    # The first trade stops using entry-open 100 and frozen ATR 1;
    # the second trade expires after exactly four exposure bars.
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
