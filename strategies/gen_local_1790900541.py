from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionAreaParams:
    area_window: int = 6
    regime_window: int = 12
    atr_window: int = 14
    area_threshold: float = 0.12
    gap_balance_threshold: float = 0.25
    trail_atr: float = 1.2
    max_hold: int = 2

    def __post_init__(self):
        if min(self.area_window, self.regime_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.0 < self.area_threshold < 1.0:
            raise ValueError('area_threshold must lie between zero and one')
        if not 0.0 < self.gap_balance_threshold < 1.0:
            raise ValueError('gap_balance_threshold must lie between zero and one')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0 or self.max_hold < 1:
            raise ValueError('Trailing multiplier and holding period must be positive')


class GeneratedStrategy(BaseStrategy[AuctionAreaParams]):
    strategy_id = 'gen_local_1790900541'

    @classmethod
    def params_type(cls):
        return AuctionAreaParams

    def warmup_bars(self, params):
        return int(params.atr_window + max(params.area_window, params.regime_window) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        scale = out['atr'].shift(1).where(lambda s: s > 0)
        out['gap'] = ((data['open'] - previous_close) / scale).replace([np.inf, -np.inf], np.nan)
        out['day_move'] = ((data['close'] - data['open']) / scale).replace([np.inf, -np.inf], np.nan)
        out['valid_step'] = out[['gap', 'day_move']].notna().all(axis=1).astype(int)
        out['gap_path'] = out['gap'].fillna(0.0).cumsum()
        out['day_path'] = out['day_move'].fillna(0.0).cumsum()
        out['cross_step'] = (
            out['gap_path'].shift(1).fillna(0.0) * out['day_move'].fillna(0.0)
            - out['day_path'].shift(1).fillna(0.0) * out['gap'].fillna(0.0)
        )
        length = params.area_window
        out['window_gap'] = out['gap'].rolling(length).sum()
        out['window_day'] = out['day_move'].rolling(length).sum()
        out['area_raw'] = (
            out['cross_step'].rolling(length).sum()
            - out['gap_path'].shift(length).fillna(0.0) * out['window_day']
            + out['day_path'].shift(length).fillna(0.0) * out['window_gap']
        )
        out['area_scale'] = (
            out['gap'].abs().rolling(length).sum()
            * out['day_move'].abs().rolling(length).sum()
        )
        out['area'] = (out['area_raw'] / out['area_scale'].where(out['area_scale'] > 0)).clip(-1.0, 1.0)
        out['regime_gap_sum'] = out['gap'].rolling(params.regime_window).sum()
        out['regime_gap_abs'] = out['gap'].abs().rolling(params.regime_window).sum()
        out['gap_balance'] = (
            out['regime_gap_sum'] / out['regime_gap_abs'].where(out['regime_gap_abs'] > 0)
        ).clip(-1.0, 1.0)
        out['valid_count'] = out['valid_step'].rolling(max(length, params.regime_window)).sum()
        out['ready'] = (
            out['valid_count'].eq(max(length, params.regime_window))
            & out[['area', 'gap_balance', 'atr']].notna().all(axis=1)
            & out['atr'].gt(0)
        )
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        trend = indicators['gap_balance'] > params.gap_balance_threshold
        balanced = indicators['gap_balance'].abs() <= params.gap_balance_threshold
        continuation = trend & (indicators['area'] > params.area_threshold) & (indicators['day_move'] > 0)
        rejection = (
            balanced & (indicators['area'] < -params.area_threshold)
            & (indicators['gap'] < 0) & (indicators['day_move'] > 0)
        )
        entry = (indicators['ready'] & (continuation | rejection)).fillna(False).to_numpy(dtype=bool)
        strength = (
            (indicators['area'].abs() - params.area_threshold)
            / (1.0 - params.area_threshold)
        ).clip(0.0, 1.0).fillna(0.0)
        proposed_size = (0.20 + 0.75 * strength).to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        active = False
        entry_bar = -1
        high_water = 0.0
        stop = 0.0
        held_size = 0.20
        for i in range(len(data)):
            if active:
                sizes[i] = held_size
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    active = False
                    continue
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or i - entry_bar >= params.max_hold:
                    active = False
                    continue
                signals[i] = 1
            elif entry[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i
                high_water = closes[i]
                stop = high_water - params.trail_atr * atrs[i]
                held_size = proposed_size[i]
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0, 103.0, 102.0, 101.0, 102.0]}, index=index)
    indicators = pd.DataFrame({
        'gap_balance': 0.5, 'area': 0.8, 'day_move': 0.1,
        'gap': 0.1, 'ready': True, 'atr': [1.0, 1.0, 10.0, 10.0, 1.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, AuctionAreaParams(max_hold=10)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    data['close'] = [100.0, 101.0, 102.0, 103.0, 104.0]
    indicators['atr'] = 1.0
    result = strategy.generate_signals(data, indicators, None, AuctionAreaParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1]
    assert result['size'].gt(0).all()
    weaker = indicators.copy()
    weaker['area'] = 0.3
    small = strategy.generate_signals(data, weaker, None, AuctionAreaParams()).data
    assert small['size'].iloc[1] < result['size'].iloc[1]


if __name__ == '__main__':
    _self_check()
