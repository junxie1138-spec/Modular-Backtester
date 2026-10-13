from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    entry_z: float = 1.5
    holding_bars: int = 8

    def __post_init__(self):
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive')
        if isinstance(self.holding_bars, bool) or not isinstance(self.holding_bars, int) or self.holding_bars < 1:
            raise ValueError('holding_bars must be a positive integer')


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = 'gen_local-w2_1791343691'

    @classmethod
    def params_type(cls) -> type[CapacityParams]:
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        volume = data['volume'].replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0)
        out['ma'] = close.rolling(20, min_periods=20).mean()
        out['price_std'] = close.rolling(20, min_periods=20).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['price_std'].where(out['price_std'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['volume_ma'] = volume.rolling(20, min_periods=20).mean().shift(1)
        out['volume_ratio'] = (volume / out['volume_ma'].where(out['volume_ma'] > 0)).replace([np.inf, -np.inf], np.nan)
        valid_z = out['z'].notna()
        downside = (out['z'] <= -params.entry_z).astype(float).where(valid_z)
        upside = (out['z'] >= params.entry_z).astype(float).where(valid_z)
        out['downside_load'] = downside.rolling(5, min_periods=5).sum()
        out['upside_load'] = upside.rolling(5, min_periods=5).sum()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        thin = indicators['volume_ratio'].between(0.0, 0.65, inclusive='right')
        long_entry = thin & (indicators['downside_load'] >= 4) & (indicators['z'] <= -params.entry_z)
        short_entry = thin & (indicators['upside_load'] >= 4) & (indicators['z'] >= params.entry_z)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if position != 0 and i == exit_bar:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
                exit_bar = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = CapacityParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['z'].isna().all()
    assert not np.isinf(derived.to_numpy()).any()
    flat = strategy.generate_signals(data, derived, None, params).data
    assert flat['signal'].eq(0).all()
    injected = pd.DataFrame({'z': [-2.0] + [2.0] * 23, 'volume_ratio': 0.5, 'downside_load': 4.0, 'upside_load': 4.0}, index=index)
    result = strategy.generate_signals(data, injected, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:9].eq(1).all()
    assert result['signal'].iloc[9] == 0
    assert result['signal'].iloc[10:18].eq(-1).all()
    assert result['signal'].iloc[18] == 0
    assert result['size'].gt(0).all()
