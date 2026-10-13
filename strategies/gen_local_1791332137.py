from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    coil_bars: int = 4
    history_bars: int = 80
    compression_percentile: float = 0.25
    density_percentile: float = 0.75
    volume_percentile: float = 0.75
    hold_bars: int = 2


def _fixed_bar_positions(entries: np.ndarray, hold_bars: int) -> np.ndarray:
    if hold_bars not in (1, 2):
        raise ValueError('hold_bars must be 1 or 2')
    positions = np.zeros(len(entries), dtype=int)
    side = 0
    exit_bar = -1
    for i, entry in enumerate(entries):
        if side:
            if i == exit_bar:
                side = 0
            else:
                positions[i] = side
        elif entry:
            side = int(entry)
            exit_bar = i + hold_bars
            positions[i] = side
    return positions


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791332137'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.coil_bars + params.history_bars + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['box_high'] = data['high'].rolling(params.coil_bars).max()
        out['box_low'] = data['low'].rolling(params.coil_bars).min()
        out['span'] = (out['box_high'] - out['box_low']) / data['close'].where(data['close'] > 0)
        out['coil_volume'] = data['volume'].rolling(params.coil_bars).mean()
        out['density'] = out['coil_volume'] / out['span'].where(out['span'] > 0)
        out['compression_threshold'] = out['span'].shift(1).rolling(params.history_bars).quantile(params.compression_percentile)
        out['density_threshold'] = out['density'].shift(1).rolling(params.history_bars).quantile(params.density_percentile)
        out['volume_threshold'] = data['volume'].shift(1).rolling(params.history_bars).quantile(params.volume_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        prior = indicators.shift(1)
        ready = prior[['box_high', 'box_low', 'span', 'density', 'compression_threshold', 'density_threshold']].notna().all(axis=1)
        loaded_coil = ready & (prior['span'] <= prior['compression_threshold']) & (prior['density'] >= prior['density_threshold'])
        volume_confirmed = indicators['volume_threshold'].notna() & (data['volume'] > indicators['volume_threshold'])
        long_entry = loaded_coil & volume_confirmed & (data['close'] > prior['box_high']) & (data['close'] > data['open'])
        short_entry = loaded_coil & volume_confirmed & (data['close'] < prior['box_low']) & (data['close'] < data['open'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = _fixed_bar_positions(entries, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    entries = np.array([1, -1, -1, -1, 1, 1, 0])
    raw = _fixed_bar_positions(entries, 2)
    shifted = pd.Series(raw).shift(1).fillna(0).astype(int).to_numpy()
    assert np.array_equal(raw, np.array([1, 1, 0, -1, -1, 0, 0]))
    assert np.array_equal(shifted, np.array([0, 1, 1, 0, -1, -1, 0]))
    assert np.array_equal(_fixed_bar_positions(np.array([1, -1, -1, 0]), 1), np.array([1, 0, -1, 0]))
