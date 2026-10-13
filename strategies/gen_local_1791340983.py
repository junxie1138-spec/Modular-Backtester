from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    entry_z: float = 1.5
    stop_atr: float = 1.5

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791340983'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous_close = close.shift(1)
        out['ma'] = close.rolling(20, min_periods=20).mean()
        out['std'] = close.rolling(20, min_periods=20).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['volume_mean'] = data['volume'].shift(1).rolling(20, min_periods=20).mean()
        out['volume_ratio'] = (data['volume'] / out['volume_mean'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['previous_volume_ratio'] = out['volume_ratio'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['entry'] = (
            out['z'].le(-params.entry_z)
            & out['previous_volume_ratio'].gt(1.8)
            & out['volume_ratio'].lt(0.8)
            & data['volume'].ge(0)
            & close.lt(previous_close)
            & out['atr'].gt(0)
            & np.isfinite(out['atr'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan

        for i in range(len(data)):
            if entry_bar >= 0:
                # The preceding close's decision fills at this bar's open.
                if i == entry_bar:
                    stop = opens[i] - params.stop_atr * entry_atr
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                if invalid or closes[i] < stop or i - entry_bar + 1 >= 4:
                    entry_bar = -1
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                entry_bar = i + 1
                entry_atr = atr[i]
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': [100, 105, 106, 107, 108, 109, 110],
        'high': [111.0] * 7,
        'low': [99.0] * 7,
        'close': [100, 110, 104, 104, 104, 104, 104],
        'volume': [1000.0] * 7,
    }, index=index)
    derived = pd.DataFrame({
        'entry': [True, False, False, False, False, False, False],
        'atr': [2.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    # Entry is 105, so the fixed stop is 102 regardless of later ATR or highs.
    data.loc[index[3], 'close'] = 101
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
