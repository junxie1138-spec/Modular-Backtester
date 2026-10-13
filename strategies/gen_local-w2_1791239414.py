from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDepthParams:
    peak_window: int = 20
    atr_window: int = 14
    depth_atr: float = 2.0
    gap_atr: float = 0.35
    trail_atr: float = 1.25
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapDepthParams]):
    strategy_id = 'gen_local-w2_1791239414'

    @classmethod
    def params_type(cls):
        return GapDepthParams

    def warmup_bars(self, params: GapDepthParams) -> int:
        return max(params.peak_window, params.atr_window) + 2

    def indicators(self, data: pd.DataFrame, params: GapDepthParams) -> pd.DataFrame:
        if min(params.peak_window, params.atr_window) < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError('Lookbacks must be >= 2 and hold_bars must be 3..5')
        if not all(np.isfinite(x) and x > 0 for x in (params.depth_atr, params.gap_atr, params.trail_atr)):
            raise ValueError('ATR multipliers must be finite and positive')
        close = data['close']
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0.0, np.nan)
        out['reference_atr'] = out['atr'].shift(1)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        out['barrier'] = out['peak'] - params.depth_atr * out['reference_atr']
        out['open_depth'] = (out['peak'] - data['open']) / out['reference_atr']
        out['close_depth'] = (out['peak'] - close) / out['reference_atr']
        out['gap'] = (data['open'] - previous) / out['reference_atr']
        out['gap_event'] = ((out['gap'] <= -params.gap_atr) & (out['open_depth'] >= params.depth_atr) & (previous >= out['barrier'])).fillna(False)
        out['reclaim'] = (out['gap_event'] & (close >= out['barrier'])).fillna(False)
        out['reject'] = (out['gap_event'] & (close < out['barrier'])).fillna(False)
        out['long_confirmed'] = (out['reclaim'].shift(1, fill_value=False) & (close >= out['barrier'].shift(1)) & (close > previous)).fillna(False)
        out['short_confirmed'] = (out['reject'].shift(1, fill_value=False) & (close < out['barrier'].shift(1)) & (close < previous)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapDepthParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_confirmed'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_confirmed'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        high_water, stop = np.nan, -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(opening[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                if not valid:
                    position = 0
                else:
                    if i == entry_bar + 1:
                        high_water = opening[i]
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # Required high-water exit also takes short profits on downward excursions.
                    if close[i] <= stop or i - entry_bar >= params.hold_bars:
                        position = 0
                signals[i] = position
                continue
            if valid and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                entry_bar = i
                high_water, stop = np.nan, -np.inf
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GapDepthParams(peak_window=3, atr_window=2, depth_atr=2.0, trail_atr=1.0)
    close = np.array([100, 100, 100, 96, 97, 98, 99, 96, 95, 94, 93, 92], dtype=float)
    opening = np.array([100, 100, 100, 94, 96, 97, 98, 94, 96, 95, 94, 93], dtype=float)
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) + 1, 'low': np.minimum(opening, close) - 1, 'close': close, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=len(close)))
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    assert ind['reclaim'].iloc[3] and ind['long_confirmed'].iloc[4]
    assert not ind['long_confirmed'].iloc[3]
    signals = strategy.generate_signals(data, ind, ctx, params).data
    assert signals['signal'].iloc[4] == 0 and signals['signal'].iloc[5] == 1
    assert signals['signal'].iloc[8] == 0
    ind.loc[data.index[7], 'atr'] = 20.0
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].iloc[8] == 0
    data.loc[data.index[3:6], 'close'] = [94.0, 93.0, 92.0]
    ind = strategy.indicators(data, params)
    assert ind['reject'].iloc[3] and ind['short_confirmed'].iloc[4]
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].iloc[5] == -1
    assert signals.index.equals(data.index) and signals['size'].gt(0).all()
    assert signals['signal'].isin([-1, 0, 1]).all()


if __name__ == '__main__':
    _self_check()
