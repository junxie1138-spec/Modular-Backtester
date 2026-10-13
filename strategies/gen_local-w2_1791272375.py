from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_bars: int = 20
    energy_bars: int = 10
    hold_bars: int = 8
    edge_fraction: float = 0.25
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791272375'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.channel_bars + 1, params.energy_bars + 2))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['change'] = data['close'].diff()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['upper'] = data['high'].shift(1).rolling(params.channel_bars).max()
        out['lower'] = data['low'].shift(1).rolling(params.channel_bars).min()
        out['width'] = (out['upper'] - out['lower']).replace(0.0, np.nan)
        out['position'] = (data['close'] - out['lower']) / out['width']
        energy = out['true_range'].pow(2)
        out['up_squared_range'] = energy.where(out['change'] > 0, 0.0).where(out['change'].notna())
        out['down_squared_range'] = energy.where(out['change'] < 0, 0.0).where(out['change'].notna())
        out['up_energy'] = out['up_squared_range'].rolling(params.energy_bars).mean()
        out['down_energy'] = out['down_squared_range'].rolling(params.energy_bars).mean()
        out['total_energy'] = (out['up_energy'] + out['down_energy']).replace(0.0, np.nan)
        out['balance'] = (out['up_energy'] - out['down_energy']) / out['total_energy']
        out['previous_balance'] = out['balance'].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars < 1 or not 0.0 < params.edge_fraction < 0.5:
            raise ValueError('hold_bars must be positive and edge_fraction must lie between zero and 0.5')
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError('sizes must satisfy 0 < min_size <= max_size <= 1')
        position = indicators['position']
        balance = indicators['balance']
        previous = indicators['previous_balance']
        long_entry = (position <= params.edge_fraction) & (balance > 0.0) & (previous <= 0.0)
        short_entry = (position >= 1.0 - params.edge_fraction) & (balance < 0.0) & (previous >= 0.0)
        ready = np.isfinite(position) & np.isfinite(balance) & np.isfinite(previous)
        entries = np.select([long_entry & ready, short_entry & ready], [1, -1], default=0)
        depth = pd.Series(np.where(entries > 0,
            (params.edge_fraction - position) / params.edge_fraction,
            (position - (1.0 - params.edge_fraction)) / params.edge_fraction), index=data.index)
        conviction = (depth.clip(0.0, 1.0) * balance.abs().clip(0.0, 1.0)).fillna(0.0)
        entry_sizes = (params.min_size + (params.max_size - params.min_size) * conviction).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        direction = 0
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if direction != 0 and i == exit_bar:
                direction = 0
                continue
            if direction == 0 and entries[i] != 0:
                direction = int(entries[i])
                exit_bar = i + params.hold_bars
                held_size = float(entry_sizes[i])
            if direction != 0:
                signals[i] = direction
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = Params()
    strategy = GeneratedStrategy()
    n = params.hold_bars
    index = pd.date_range('2020-01-01', periods=2 * n + 4)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'position': 0.5, 'balance': 0.0, 'previous_balance': 0.0}, index=index)
    ind.iloc[0] = [0.1, 0.2, -0.1]
    ind.iloc[2] = [0.99, -0.9, 0.1]
    ind.iloc[n + 1] = [0.99, -0.9, 0.1]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:n + 1] == 1).all()
    assert result['signal'].iloc[n + 1] == 0
    assert (result['signal'].iloc[n + 2:2 * n + 2] == -1).all()
    assert result['signal'].iloc[2 * n + 2] == 0
    assert result['size'].iloc[n + 2] > result['size'].iloc[1]
    assert (result['size'] > 0.0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert (flat['signal'] == 0).all()


if __name__ == '__main__':
    _self_check()
