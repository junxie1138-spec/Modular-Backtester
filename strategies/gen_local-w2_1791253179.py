from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 24
    atr_multiple: float = 3.0

    def __post_init__(self):
        if not isinstance(self.graph_window, (int, np.integer)) or self.graph_window < 8:
            raise ValueError('graph_window must be an integer >= 8')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be positive and finite')


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1791253179'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(params.graph_window + 1, 15)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        close = data['close'].astype(float)
        window = params.graph_window
        bound = np.full(len(data), np.inf)
        degree = np.zeros(len(data), dtype=float)
        distant_up = np.zeros(len(data), dtype=float)
        # A past endpoint is visible when its slope is below every nearer slope.
        for lag in range(1, window + 1):
            slope = ((close - close.shift(lag)) / lag).to_numpy()
            finite = np.isfinite(slope)
            visible = finite & (slope < bound)
            degree += visible
            if lag > window // 2:
                distant_up += visible & (slope > 0)
            bound = np.minimum(bound, np.where(finite, slope, np.inf))
        complete = close.where(np.isfinite(close)).rolling(window + 1).count().eq(window + 1)
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out['degree'] = pd.Series(degree, index=data.index).where(complete)
        out['distant_up_degree'] = pd.Series(distant_up, index=data.index).where(complete)
        out['distant_up_share'] = out['distant_up_degree'] / out['degree'].replace(0, np.nan)
        out['momentum'] = (close - close.shift(window)).where(complete)
        out['atr'] = true_range.where(np.isfinite(true_range)).rolling(14, min_periods=14).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        entry = (
            indicators['degree'].ge(4)
            & indicators['distant_up_degree'].ge(3)
            & indicators['distant_up_share'].ge(0.4)
            & indicators['momentum'].gt(0)
            & indicators['atr'].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        held = False
        age = 0
        cooldown = 0
        high_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if held:
                age += 1
                stopped = False
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.atr_multiple * atr[i])
                    stopped = close[i] <= stop
                if stopped or age >= 18:
                    held = False
                    cooldown = 3
            elif cooldown > 0:
                cooldown -= 1
            elif valid and entry[i]:
                held = True
                age = 0
                high_water = close[i]
                stop = high_water - params.atr_multiple * atr[i]
            signal[i] = int(held)
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=90, freq='B')
    close = 100.0 + 0.02 * np.arange(90) ** 2
    data = pd.DataFrame({'open': close, 'high': close + 0.5,
                         'low': close - 0.5, 'close': close,
                         'volume': 1000000.0}, index=index)
    ind = strategy.indicators(data, params)
    assert ind['degree'].iloc[params.graph_window:].eq(params.graph_window).all()
    assert ind['distant_up_degree'].iloc[params.graph_window:].eq(12).all()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].iloc[:25].eq(0).all()
    assert result['signal'].iloc[25:43].eq(1).all()
    assert result['signal'].iloc[43] == 0
    falling = data.copy()
    falling.iloc[30, falling.columns.get_loc('close')] = 1.0
    stopped = strategy.generate_signals(falling, ind, None, params).data
    assert stopped['signal'].iloc[30] == 1
    assert stopped['signal'].iloc[31] == 0
    assert stopped['signal'].isin([0, 1]).all()
    assert stopped['size'].gt(0).all()
