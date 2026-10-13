from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    fast_bars: int = 3
    slow_bars: int = 20
    roc_bars: int = 3
    atr_bars: int = 14
    compression_ratio: float = 0.70
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local_1791199839'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        if not (1 <= params.fast_bars < params.slow_bars):
            raise ValueError('Require 1 <= fast_bars < slow_bars')
        if params.roc_bars < 1 or params.atr_bars < 1:
            raise ValueError('Lookbacks must be positive')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')
        if not (0 < params.compression_ratio < 1):
            raise ValueError('compression_ratio must be between 0 and 1')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be positive and finite')
        # One extra bar for confirmation after the longest derived window.
        return int(max(params.slow_bars + 1, params.roc_bars + 2,
                       params.atr_bars + 1) + 1)

    def indicators(self, data: pd.DataFrame,
                   params: CompressionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        denominator = out['previous_close'].where(out['previous_close'] > 0)
        out['daily_range'] = (data['high'] - data['low']).clip(lower=0)
        out['fractional_range'] = out['daily_range'] / denominator
        out['fast_range'] = out['fractional_range'].rolling(
            params.fast_bars, min_periods=params.fast_bars).mean()
        out['slow_range'] = out['fractional_range'].rolling(
            params.slow_bars, min_periods=params.slow_bars).mean()
        out['compression'] = out['fast_range'] / out['slow_range'].where(
            out['slow_range'] > 0)
        out['roc'] = data['close'].pct_change(
            periods=params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['daily_range', 'high_gap', 'low_gap']].max(
            axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['eligible'] = (
            (out['compression'] < params.compression_ratio)
            & (out['roc'] < 0)
            & (out['acceleration'] > 0)
            & (out['atr'] > 0)
        ).fillna(False)
        out['confirmed'] = (
            out['eligible'] & out['eligible'].shift(1, fill_value=False)
        )
        # Each uninterrupted confirmation episode permits only one entry.
        out['entry'] = (
            out['confirmed'] & ~out['confirmed'].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: CompressionParams) -> SignalFrame:
        self.warmup_bars(params)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        entry_atr = np.nan
        stop = np.nan

        for i in range(len(data)):
            if active:
                if age == 0:
                    # This opening price is observed only on the fill bar.
                    # ATR was frozen at the preceding close's entry decision.
                    stop = opens[i] - params.stop_atr * entry_atr
                age += 1
                if (not np.isfinite(stop)
                        or not np.isfinite(closes[i])
                        or closes[i] < stop
                        or age >= params.hold_bars):
                    active = False
                else:
                    raw[i] = 1
            elif (entries[i] and np.isfinite(closes[i])
                  and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                active = True
                age = 0
                entry_atr = atr[i]
                raw[i] = 1

        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 97.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        'entry': [False, True, False, False, False], 'atr': 1.0,
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert (result['size'] > 0).all()
    data.loc[index[2], 'close'] = 98.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0]
    data.loc[index[2], 'close'] = 100.0
    result = strategy.generate_signals(
        data, derived, None, CompressionParams(hold_bars=1)).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0]
    # Synthetic contracting decline: confirmation must precede entry.
    index = pd.date_range('2020-02-01', periods=6)
    close = np.array([100.0, 99.0, 98.0, 97.4, 97.1, 97.0])
    width = np.array([4.0, 4.0, 4.0, 1.0, 0.6, 0.4])
    sample = pd.DataFrame({
        'open': close, 'high': close + width / 2,
        'low': close - width / 2, 'close': close, 'volume': 1000.0,
    }, index=index)
    small = CompressionParams(fast_bars=1, slow_bars=3,
                              roc_bars=1, atr_bars=2,
                              compression_ratio=0.9)
    computed = strategy.indicators(sample, small)
    assert computed['entry'].tolist() == [False, False, False, False, True, False]
    assert strategy.warmup_bars(small) >= 5


if __name__ == '__main__':
    _self_check()
