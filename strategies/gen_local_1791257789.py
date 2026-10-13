from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    peak_window: int = 20
    shelf_window: int = 3
    percentile_window: int = 63
    atr_window: int = 14
    depth_quantile: float = 0.75
    compression_quantile: float = 0.25
    breakeven_gain: float = 0.006
    trail_atr: float = 1.8
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = 'gen_local_1791257789'

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    def warmup_bars(self, params: ShelfParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1) + params.shelf_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        if min(params.peak_window, params.shelf_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not (0 < params.depth_quantile < 1 and 0 < params.compression_quantile < 1):
            raise ValueError('Quantiles must lie strictly between zero and one.')
        if params.trail_atr <= 0 or params.breakeven_gain <= 0 or not 3 <= params.max_hold <= 5:
            raise ValueError('Positive stop parameters and a three-to-five-session hold are required.')
        out = pd.DataFrame(index=data.index)
        peak = data['close'].rolling(params.peak_window).max().replace(0, np.nan)
        out['drawdown'] = (1.0 - data['close'] / peak).clip(lower=0)
        out['depth_floor'] = out['drawdown'].rolling(params.shelf_window).min()
        out['depth_span'] = (out['drawdown'].rolling(params.shelf_window).max() - out['depth_floor'])
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['range_ratio'] = out['true_range'].rolling(params.shelf_window).mean() / out['atr'].replace(0, np.nan)
        out['depth_threshold'] = out['depth_floor'].rolling(params.percentile_window).quantile(params.depth_quantile).shift(1)
        out['span_threshold'] = out['depth_span'].rolling(params.percentile_window).quantile(params.compression_quantile).shift(1)
        out['range_threshold'] = out['range_ratio'].rolling(params.percentile_window).quantile(params.compression_quantile).shift(1)
        ready = out.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        eligible = (
            ready
            & (out['depth_floor'] > out['depth_threshold'])
            & (out['depth_span'] <= out['span_threshold'])
            & (out['range_ratio'] <= out['range_threshold'])
            & (out['atr'] > 0)
        )
        out['entry_event'] = eligible & ~eligible.shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShelfParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i == 0 or not np.isfinite(opening[i]) or opening[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                # The preceding decision fills at this bar's open.
                entry = opening[i]
                stop = entry - params.trail_atr * atr[i - 1]
                active, armed, age = True, False, 0
            if active:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # Close-confirmed exits become flat at the following open.
                if close[i] <= stop or age >= params.max_hold:
                    active = False
                    continue
                armed = armed or close[i] >= entry * (1.0 + params.breakeven_gain)
                if armed:
                    stop = max(stop, entry, close[i] - params.trail_atr * atr[i])
                raw[i] = 1
            elif entries[i] and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({'signal': raw, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': [100, 101, 103, 102, 101.1, 100, 100, 100], 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'atr': [1, 1, 1, 10, 1, 1, 1, 1], 'entry_event': [True, False, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, ShelfParams(max_hold=5)).data
    # Rising ATR cannot loosen the stop; the fourth close breaches it.
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data['close'] = [100, 101, 100, 100, 100, 100, 100, 100]
    ind['atr'] = 1.0
    result = strategy.generate_signals(data, ind, None, ShelfParams(max_hold=5)).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    data['close'] = 100.0
    result = strategy.generate_signals(data, ind, None, ShelfParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.indicators(data, ShelfParams())['entry_event'].eq(False).all()


if __name__ == '__main__':
    _self_check()
