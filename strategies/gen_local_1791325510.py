from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_order: float = 0.4
    fd_window: int = 40
    z_window: int = 60
    entry_z: float = 1.5
    vol_window: int = 20
    target_vol: float = 0.12
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791325510'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fd_window + params.z_window, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0 < params.fd_order < 1 and params.fd_window >= 2 and params.z_window >= 2 and params.vol_window >= 2 and params.entry_z > 0 and params.target_vol > 0 and params.holding_bars in (1, 2)):
            raise ValueError('Invalid fractional-difference, volatility, or holding parameters')
        close = data['close'].astype(float).where(lambda x: x > 0)
        weights = np.empty(params.fd_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.fd_window):
            weights[k] = -weights[k - 1] * (params.fd_order - k + 1) / k
        # ponytail: fixed-width fractional memory; enlarge fd_window for longer memory.
        fd = close.copy()
        for k in range(1, params.fd_window):
            fd = fd + weights[k] * close.shift(k)
        mean = fd.rolling(params.z_window, min_periods=params.z_window).mean()
        deviation = fd.rolling(params.z_window, min_periods=params.z_window).std(ddof=0).replace(0.0, np.nan)
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fd
        out['fractional_mean'] = mean
        out['fractional_std'] = deviation
        out['z'] = (fd - mean) / deviation
        out['return'] = returns
        out['volatility'] = volatility
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['entry_size'] = (params.target_vol / volatility.replace(0.0, np.nan)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError('holding_bars must be 1 or 2')
        z = indicators['z']
        previous = z.shift(1)
        valid = indicators[['z', 'ma200', 'entry_size']].notna().all(axis=1) & previous.notna()
        long_entry = valid & (data['close'] > indicators['ma200']) & (previous < -params.entry_z) & (z >= -params.entry_z) & (z < 0)
        short_entry = valid & (data['close'] < indicators['ma200']) & (previous > params.entry_z) & (z <= params.entry_z) & (z > 0)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        entry_sizes = indicators['entry_size'].fillna(0.01).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        position = 0
        exit_bar = -1
        locked_size = 0.01
        for i in range(len(data)):
            if position and i == exit_bar:
                position = 0
                continue
            if position == 0 and candidate[i] != 0:
                position = int(candidate[i])
                exit_bar = i + params.holding_bars
                locked_size = float(entry_sizes[i])
            signal[i] = position
            if position:
                sizes[i] = locked_size
        df = pd.DataFrame({'signal': signal, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': 101.0, 'high': 102.0, 'low': 100.0, 'close': 101.0, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'z': [-2.0, -1.0, -2.0, -1.0, 2.0, 1.0, 2.0, 1.0, 0.0], 'ma200': [100.0] * 4 + [102.0] * 5, 'entry_size': [0.3, 0.4, 0.8, 0.9, 0.2, 0.5, 0.7, 0.8, 0.9]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, Params()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert np.allclose(result.loc[result['signal'] != 0, 'size'], [0.4, 0.4, 0.5, 0.5])
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(Params()) >= 200
