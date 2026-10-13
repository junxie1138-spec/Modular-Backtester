from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 30
    short_window: int = 5
    long_window: int = 40
    compression_ratio: float = 0.30
    entry_z: float = 0.80
    hold_bars: int = 8
    size: float = 0.95

    def __post_init__(self):
        if self.ma_window < 2 or self.short_window < 2 or self.long_window <= self.short_window:
            raise ValueError('Require ma_window >= 2 and 2 <= short_window < long_window')
        if not 0.0 < self.compression_ratio < 1.0:
            raise ValueError('compression_ratio must be between zero and one')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0.0:
            raise ValueError('entry_z must be finite and positive')
        if self.hold_bars < 1 or not 0.0 < self.size <= 1.0:
            raise ValueError('Require positive hold_bars and size in (0, 1]')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791319091'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ma_window, params.long_window, params.short_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ma'] = data['close'].rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['std'] = data['close'].rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = ((data['close'] - out['ma']) / out['std'].where(out['std'] > 0.0)).replace([np.inf, -np.inf], np.nan)
        out['short_high'] = data['high'].rolling(params.short_window, min_periods=params.short_window).max()
        out['short_low'] = data['low'].rolling(params.short_window, min_periods=params.short_window).min()
        out['long_high'] = data['high'].rolling(params.long_window, min_periods=params.long_window).max()
        out['long_low'] = data['low'].rolling(params.long_window, min_periods=params.long_window).min()
        out['short_width'] = out['short_high'] - out['short_low']
        out['long_width'] = out['long_high'] - out['long_low']
        out['compression'] = (out['short_width'] / out['long_width'].where(out['long_width'] > 0.0)).replace([np.inf, -np.inf], np.nan)
        out['eligible'] = ((out['short_width'] >= 0.0) & (out['compression'] <= params.compression_ratio) & (out['z'] <= -params.entry_z)).fillna(False)
        out['confirmed'] = out['eligible'] & out['eligible'].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        position = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    position[i] = 1
                else:
                    exit_bar = -1
            elif entry:
                position[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({'signal': position, 'size': float(params.size)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params()
    index = pd.date_range('2020-01-01', periods=20, freq='B')
    data = pd.DataFrame(index=index)
    eligible = pd.Series([False, True, False, True, True] + [True] * 15, index=index)
    indicators = pd.DataFrame({'confirmed': eligible & eligible.shift(1, fill_value=False)}, index=index)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].iloc[:5].eq(0).all()
    assert signals['signal'].iloc[5:13].eq(1).all()
    assert signals['signal'].iloc[13] == 0
    assert signals['signal'].iloc[14] == 1
    assert signals['size'].gt(0.0).all()
