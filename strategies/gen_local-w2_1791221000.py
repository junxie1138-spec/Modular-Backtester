from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    peak_window: int = 40
    atr_window: int = 14
    shape_bars: int = 6
    history_bars: int = 504
    holding_bars: int = 8
    neighbors: int = 8
    min_depth_atr: float = 1.0
    min_repayment: float = 0.30
    trail_atr: float = 2.0

    def __post_init__(self):
        lengths = (self.peak_window, self.atr_window, self.shape_bars,
                   self.history_bars, self.holding_bars, self.neighbors)
        if any(type(x) is not int or x < 1 for x in lengths):
            raise ValueError('Lookbacks and counts must be positive integers')
        if self.shape_bars < 2 or not 5 <= self.holding_bars <= 10:
            raise ValueError('Use at least two shape bars and a 5-10 bar holding horizon')
        if self.history_bars // self.holding_bars < self.neighbors:
            raise ValueError('History must accommodate the requested analog count')
        if any(not np.isfinite(x) or x <= 0 for x in
               (self.min_depth_atr, self.min_repayment, self.trail_atr)):
            raise ValueError('Thresholds must be finite and positive')


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791221000'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1)
                   + params.shape_bars + params.holding_bars + params.history_bars)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out['peak'] = close.rolling(p.peak_window, min_periods=p.peak_window).max()
        out['depth'] = out['peak'] - close
        out['depth_atr'] = out['depth'] / out['atr'].where(out['atr'] > 0)
        shape_columns = []
        for lag in range(p.shape_bars):
            name = f'depth_lag_{lag}'
            out[name] = out['depth_atr'].shift(lag)
            shape_columns.append(name)
        shapes = out[shape_columns].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        depths = out['depth'].to_numpy(dtype=float)
        repayment = np.full(len(data), np.nan)
        win_rate = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        # ponytail: bounded brute-force search; index shapes if history grows substantially.
        for t in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(shapes[t]).all() or shapes[t, 0] < p.min_depth_atr:
                continue
            end = t - p.shape_bars - p.holding_bars + 1
            # Labels finish before the current shape; label intervals do not overlap.
            candidates = np.arange(end - p.history_bars, end, p.holding_bars)
            valid = (np.isfinite(shapes[candidates]).all(axis=1)
                     & (shapes[candidates, 0] >= p.min_depth_atr)
                     & np.isfinite(prices[candidates + p.holding_bars])
                     & (depths[candidates] > 0))
            candidates = candidates[valid]
            if len(candidates) < p.neighbors:
                continue
            errors = np.mean((shapes[candidates] - shapes[t]) ** 2, axis=1)
            nearest = np.argsort(errors, kind='stable')[:p.neighbors]
            selected = candidates[nearest]
            outcomes = ((prices[selected + p.holding_bars] - prices[selected])
                        / depths[selected])
            repayment[t] = np.median(outcomes)
            win_rate[t] = np.mean(outcomes > 0)
            distance[t] = np.sqrt(np.mean(errors[nearest]))
        out['analog_repayment'] = repayment
        out['analog_win_rate'] = win_rate
        out['analog_distance'] = distance
        out['eligible'] = (
            (out['analog_repayment'] >= p.min_repayment)
            & (out['analog_win_rate'] >= 0.625)
            & (out['analog_distance'] <= 0.75)
            & (out['depth_atr'] >= p.min_depth_atr)
            & (out['depth_atr'] < out['depth_atr'].shift(1))
            & (close > out['previous_close'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        eligible = indicators['eligible'].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        target = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
                active = False
                high_water = stop = -np.inf
                continue
            if active:
                # Entry executes on entry_bar; its first close starts the high-water mark.
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.trail_atr * atr)
                if price <= stop or i - entry_bar + 1 >= params.holding_bars:
                    active = False
                    high_water = stop = -np.inf
                else:
                    target[i] = 1
            elif confirmed[i]:
                active = True
                entry_bar = i + 1
                high_water = stop = -np.inf
                target[i] = 1
        df = pd.DataFrame({'signal': target, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    p = AnalogParams()
    index = pd.date_range('2020-01-01', periods=11)
    data = pd.DataFrame({'close': [100, 101, 102, 105, 104, 102, 101, 101, 101, 101, 101]},
                        index=index)
    supplied = pd.DataFrame({
        'eligible': [True, True] + [False] * 9,
        'atr': [1.0] * 4 + [10.0] * 7
    }, index=index)
    signals = strategy.generate_signals(data, supplied, None, p).data
    assert signals['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    supplied['atr'] = 10.0
    assert strategy.generate_signals(data, supplied, None, p).data['signal'].tolist() == (
        [0, 0] + [1] * 8 + [0])
    n = strategy.warmup_bars(p) + 90
    x = np.arange(n, dtype=float)
    close = 100 + 0.02 * x + 4 * np.sin(x / 11)
    history = pd.DataFrame({'open': close, 'high': close + 1,
                            'low': close - 1, 'close': close, 'volume': 100000.0},
                           index=pd.date_range('2018-01-01', periods=n))
    cut = n - 30
    full = strategy.indicators(history, p)
    prefix = strategy.indicators(history.iloc[:cut], p)
    pd.testing.assert_frame_equal(full.iloc[:cut], prefix)
    pd.testing.assert_frame_equal(
        strategy.generate_signals(history, full, None, p).data.iloc[:cut],
        strategy.generate_signals(history.iloc[:cut], prefix, None, p).data)
    assert not full['eligible'].iloc[:strategy.warmup_bars(p) - 1].any()


if __name__ == '__main__':
    _self_check()
