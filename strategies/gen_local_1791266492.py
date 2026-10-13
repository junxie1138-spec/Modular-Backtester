from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    compression_bars: int = 4
    baseline_bars: int = 20
    roc_bars: int = 3
    compression_ratio: float = 0.75
    acceleration_threshold: float = 0.20
    hold_bars: int = 2


def _fixed_bar_positions(entries: np.ndarray, hold_bars: int) -> np.ndarray:
    if hold_bars not in (1, 2):
        raise ValueError('hold_bars must be 1 or 2')
    positions = np.zeros(len(entries), dtype=int)
    exit_at = -1
    active = False
    for i, entry in enumerate(entries):
        if active:
            if i < exit_at:
                positions[i] = 1
            else:
                active = False
        elif entry:
            positions[i] = 1
            exit_at = i + hold_bars
            active = True
    return positions


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = 'gen_local_1791266492'

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(200, params.baseline_bars + 2,
                       params.compression_bars + 2, params.roc_bars + 3))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.compression_bars, params.baseline_bars, params.roc_bars) < 1:
            raise ValueError('Lookbacks must be positive')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['baseline_range'] = out['true_range'].rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean().shift(1)
        out['compressed_range'] = out['true_range'].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).mean().shift(1)
        out['compression'] = out['compressed_range'] / out['baseline_range'].replace(0, np.nan)
        out['box_high'] = data['high'].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).max().shift(1)
        out['box_low'] = data['low'].rolling(
            params.compression_bars, min_periods=params.compression_bars
        ).min().shift(1)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['roc'] = data['close'].pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff()
        out['previous_acceleration'] = out['acceleration'].shift(1)
        out['range_fraction'] = out['baseline_range'] / out['previous_close'].replace(0, np.nan)
        out['scaled_acceleration'] = out['acceleration'] / out['range_fraction'].replace(0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            out.notna().all(axis=1)
            & (data['close'] > out['ma200'])
            & (out['compression'] < params.compression_ratio)
            & (out['roc'] < 0)
            & (out['previous_acceleration'] <= 0)
            & (out['scaled_acceleration'] > params.acceleration_threshold)
            & (data['close'] >= out['box_low'])
            & (data['close'] <= out['box_high'])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: CompressionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        df['signal'] = _fixed_bar_positions(entries, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    entries = np.ones(8, dtype=bool)
    raw = _fixed_bar_positions(entries, 2)
    shifted = pd.Series(raw).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert _fixed_bar_positions(entries, 1).tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
