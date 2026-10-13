from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entry_z: float = 1.5
    atr_multiple: float = 1.25

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0.25:
            raise ValueError('entry_z must be finite and greater than 0.25')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791210499'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 20

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float)
        volume = data['volume'].astype(float).where(data['volume'] >= 0)
        ma = close.rolling(20, min_periods=20).mean()
        deviation = close.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
        volume_sum = volume.rolling(20, min_periods=20).sum().replace(0, np.nan)
        vwma = (close * volume).rolling(20, min_periods=20).sum() / volume_sum
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        result = pd.DataFrame({
            'ma': ma,
            'deviation': deviation,
            'volume_sum': volume_sum,
            'vwma': vwma,
            'volume_premium': vwma - ma,
            'z': (close - ma) / deviation,
            'previous_close': previous_close,
            'true_range': true_range,
            'atr': true_range.rolling(14, min_periods=14).mean(),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        age = 0
        highest_close = np.nan
        stop = -np.inf
        values = indicators[['z', 'volume_premium', 'atr']].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)

        for i, (z, premium, atr) in enumerate(values):
            close = closes[i]
            valid = np.isfinite([close, z, premium, atr]).all() and close > 0 and atr > 0
            if not valid:
                active = False
                highest_close = np.nan
                stop = -np.inf
                age = 0
                continue

            if z >= -0.25:
                armed = True

            if active:
                # Entry fills on this bar after the final signal shift.
                age += 1
                highest_close = close if np.isnan(highest_close) else max(highest_close, close)
                stop = max(stop, highest_close - params.atr_multiple * atr)
                if close <= stop or z >= -0.25 or age >= 2:
                    active = False
                    highest_close = np.nan
                    stop = -np.inf
                    age = 0
                else:
                    signals[i] = 1
                continue

            if armed and z <= -params.entry_z and premium > 0:
                active = True
                armed = False
                age = 0
                highest_close = np.nan
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [103.0] * 6,
        'low': [98.0] * 6,
        'close': [100.0, 102.0, 99.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'z': [-2.0, -2.0, -2.0, 0.0, -2.0, -2.0],
        'volume_premium': [1.0] * 6,
        'atr': [1.0] * 6,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1]
    assert result['size'].gt(0).all()
    assert result.index.equals(data.index)
    indicators['volume_premium'] = -1.0
    assert strategy.generate_signals(data, indicators, None, Params()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
