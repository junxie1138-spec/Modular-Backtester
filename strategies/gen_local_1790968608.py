from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_bars: int = 20
    volume_bars: int = 20
    atr_bars: int = 14
    volume_multiple: float = 1.6
    edge_fraction: float = 0.2
    max_progress_atr: float = 0.35
    rejection_fraction: float = 0.65
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790968608'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.range_bars + 1,
                       params.volume_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.range_bars, params.volume_bars, params.atr_bars) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not (3 <= params.hold_bars <= 5):
            raise ValueError('Holding period must be three to five sessions.')
        if not (0 < params.edge_fraction < 0.5
                and 0.5 < params.rejection_fraction < 1
                and params.volume_multiple > 0
                and params.max_progress_atr > 0
                and params.stop_atr > 0):
            raise ValueError('Invalid signal or stop parameters.')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['range_low'] = data['low'].shift(1).rolling(params.range_bars).min()
        out['range_high'] = data['high'].shift(1).rolling(params.range_bars).max()
        out['range_width'] = (out['range_high'] - out['range_low']).where(
            out['range_high'] > out['range_low'])
        out['low_position'] = (data['low'] - out['range_low']) / out['range_width']
        out['high_position'] = (data['high'] - out['range_low']) / out['range_width']
        out['volume_baseline'] = data['volume'].shift(1).rolling(params.volume_bars).mean()
        out['relative_volume'] = data['volume'] / out['volume_baseline'].where(
            out['volume_baseline'] > 0)
        out['daily_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_location'] = (data['close'] - data['low']) / out['daily_range']
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out['progress_atr'] = (data['close'] - out['previous_close']) / out['atr'].where(out['atr'] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (out['atr'] > 0)
        busy = out['relative_volume'] >= params.volume_multiple
        long_entry = (
            ready & busy & (data['close'] > out['ma200'])
            & (out['low_position'] <= params.edge_fraction)
            & (out['progress_atr'] < 0)
            & (out['progress_atr'] >= -params.max_progress_atr)
            & (out['close_location'] >= params.rejection_fraction)
        )
        short_entry = (
            ready & busy & (data['close'] < out['ma200'])
            & (out['high_position'] >= 1 - params.edge_fraction)
            & (out['progress_atr'] > 0)
            & (out['progress_atr'] <= params.max_progress_atr)
            & (out['close_location'] <= 1 - params.rejection_fraction)
        )
        out['candidate'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        side = 0
        age = 0
        stop = np.nan
        distance = np.nan
        for i in range(len(data)):
            if side:
                if age == 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    # Fixed anchor: entry-session open, with configured 5 bps slippage.
                    entry = opens[i] * (1.0 + side * 0.0005)
                    stop = entry - side * distance
                age += 1
                stopped = np.isfinite(closes[i]) and side * (closes[i] - stop) < 0
                if stopped or age >= params.hold_bars or not np.isfinite(closes[i]):
                    side = 0
                    continue
                signals[i] = side
            elif candidates[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                distance = params.stop_atr * atr[i]
                age = 0
                signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'close': 100.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'candidate': 0}, index=index)
    indicators.iloc[0, 1] = 1
    indicators.iloc[5, 1] = -1
    data.iloc[2, 1] = 98.0
    data.iloc[7, 1] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, -1, -1, 0, 0]
    data['close'] = 100.0
    data.iloc[1, 1] = 104.0
    indicators.iloc[5, 1] = 0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
