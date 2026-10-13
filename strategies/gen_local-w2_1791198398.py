from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CorridorParams:
    channel_window: int = 12
    overlap_window: int = 3
    atr_window: int = 14
    min_overlap: float = 0.35
    channel_ceiling: float = 0.65
    stop_atr: float = 1.3
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CorridorParams]):
    strategy_id = 'gen_local-w2_1791198398'

    @classmethod
    def params_type(cls) -> type[CorridorParams]:
        return CorridorParams

    def warmup_bars(self, params: CorridorParams) -> int:
        return int(max(params.channel_window, params.overlap_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CorridorParams) -> pd.DataFrame:
        if min(params.channel_window, params.atr_window) < 1 or params.overlap_window < 2:
            raise ValueError('Invalid lookback windows')
        if not (0 < params.min_overlap < 1 and 0 < params.channel_ceiling <= 1):
            raise ValueError('Invalid geometry thresholds')
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0 and 3 <= params.hold_bars <= 5):
            raise ValueError('Invalid stop or holding horizon')
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError('Invalid position sizes')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['channel_low'] = data['low'].rolling(params.channel_window).min()
        out['channel_high'] = data['high'].rolling(params.channel_window).max()
        out['channel_width'] = (out['channel_high'] - out['channel_low']).where(lambda x: x > 0)
        out['channel_position'] = (data['close'] - out['channel_low']) / out['channel_width']
        out['corridor_low'] = data['low'].rolling(params.overlap_window).max()
        out['corridor_high'] = data['high'].rolling(params.overlap_window).min()
        out['corridor_width'] = (out['corridor_high'] - out['corridor_low']).where(lambda x: x > 0)
        out['union_low'] = data['low'].rolling(params.overlap_window).min()
        out['union_high'] = data['high'].rolling(params.overlap_window).max()
        out['union_width'] = (out['union_high'] - out['union_low']).where(lambda x: x > 0)
        out['overlap_fraction'] = out['corridor_width'] / out['union_width']
        out['corridor_position'] = (data['close'] - out['corridor_low']) / out['corridor_width']
        out['previous_corridor_position'] = out['corridor_position'].shift(1)
        out['reclaim'] = (out['previous_corridor_position'] <= 0.5) & (out['corridor_position'] > 0.5)
        out['strength'] = ((2 * out['corridor_position'] - 1).clip(0, 1) * out['overlap_fraction'].clip(0, 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['entry'] = (
            out['reclaim']
            & (out['corridor_position'] <= 1.0)
            & (out['overlap_fraction'] >= params.min_overlap)
            & out['channel_position'].between(0.0, params.channel_ceiling)
            & (out['atr'] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CorridorParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].fillna(0.0).clip(0, 1).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        entry_bar = -1
        peak = -np.inf
        stop = -np.inf
        entry_size = params.min_size
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                expired = i - entry_bar >= params.hold_bars
                if valid:
                    # Entry executes here or earlier, after the decision bar.
                    peak = max(peak, closes[i])
                    stop = max(stop, peak - params.stop_atr * atrs[i])
                if not valid or expired or closes[i] <= stop:
                    active = False
                    peak = stop = -np.inf
                else:
                    signals[i] = 1
                    sizes[i] = entry_size
            elif valid and entries[i]:
                active = True
                entry_bar = i
                peak = stop = -np.inf
                entry_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    close = np.array([100, 102, 104, 103, 100, 100, 100, 100], dtype=float)
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
    params = CorridorParams(hold_bars=5)
    strategy = GeneratedStrategy()
    ind = pd.DataFrame({'atr': [1, 1, 1, 10, 10, 10, 10, 10], 'entry': [True, False, False, False, False, False, False, False], 'strength': 0.8}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.isclose(result['size'].iloc[1], 0.81)
    assert (result['size'] > 0).all()
    ind['atr'] = 100.0
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 1, 0, 0]
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
