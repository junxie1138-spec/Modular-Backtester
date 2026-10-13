from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 5
    reference_bars: int = 63
    atr_bars: int = 14
    compression_percentile: float = 0.30
    area_percentile: float = 0.80
    stop_atr: float = 1.50
    hold_bars: int = 2

    def __post_init__(self):
        if self.path_bars < 2 or self.reference_bars < 2 or self.atr_bars < 1:
            raise ValueError('Invalid lookback lengths')
        if not 0 < self.compression_percentile < 1 or not 0 < self.area_percentile < 1:
            raise ValueError('Percentiles must lie between zero and one')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or self.hold_bars not in (1, 2):
            raise ValueError('Invalid stop or holding period')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791269686'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(params.path_bars + params.reference_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['log_return'] = np.log(close).diff()
        m = params.path_bars
        out['displacement'] = out['log_return'].rolling(m, min_periods=m).sum()
        out['variation'] = out['log_return'].abs().rolling(m, min_periods=m).sum()
        # Antisymmetric second-order signature for unit time steps and log-price increments.
        area = pd.Series(0.0, index=data.index)
        for lag in range(m):
            area = area + 0.5 * (m - 1 - 2 * lag) * out['log_return'].shift(lag)
        out['signature_area'] = area
        out['normalized_area'] = area / (m * out['variation'].where(out['variation'] > 0))
        out['channel_high'] = data['high'].rolling(m, min_periods=m).max()
        out['channel_low'] = data['low'].rolling(m, min_periods=m).min()
        out['compression'] = (out['channel_high'] - out['channel_low']) / close
        n = params.reference_bars
        out['compression_threshold'] = out['compression'].shift(1).rolling(n, min_periods=n).quantile(params.compression_percentile)
        out['area_threshold'] = out['normalized_area'].shift(1).rolling(n, min_periods=n).quantile(params.area_percentile)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['setup'] = (
            (out['compression'] < out['compression_threshold'])
            & (out['normalized_area'] > out['area_threshold'])
            & (out['normalized_area'] > 0)
            & (out['displacement'] < 0)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        setup = indicators['setup'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    # Entry executes at this open; freeze the ATR known at the preceding close.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    age = 0
                    active = True
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif setup[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [101.0] * 6,
        'low': [96.0] * 6,
        'close': [100.0, 97.0, 100.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({'setup': [True, False, True, False, False, False], 'atr': [1.0] * 6}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, SignatureParams()).data
    # First trade stops after one session; second expires after two. Both enter one bar later.
    assert result['signal'].tolist() == [0, 1, 0, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
