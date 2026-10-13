from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    complexity_window: int = 18
    complexity_floor: float = 0.80
    breakout_window: int = 12
    breakout_atr: float = 0.15
    atr_window: int = 14
    trail_atr: float = 1.50
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.complexity_window, self.breakout_window, self.atr_window) < 3:
            raise ValueError('Lookbacks must be at least three bars')
        if not 0.0 <= self.complexity_floor <= 1.0:
            raise ValueError('Complexity floor must be between zero and one')
        if not np.isfinite(self.breakout_atr) or self.breakout_atr < 0.0:
            raise ValueError('Breakout distance must be finite and nonnegative')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0.0:
            raise ValueError('Trailing distance must be finite and positive')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('Holding horizon must be three to five sessions')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791217472'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.complexity_window + 3,
                       params.breakout_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close'].astype(float)
        oldest = close.shift(2)
        middle = close.shift(1)
        triplet_valid = (
            np.isfinite(oldest) & np.isfinite(middle) & np.isfinite(close)
            & oldest.ne(middle) & oldest.ne(close) & middle.ne(close)
        )
        code = ((middle > oldest).astype(int)
                + 2 * (close > middle).astype(int)
                + 4 * (close > oldest).astype(int)).where(triplet_valid)
        out = pd.DataFrame(index=data.index)
        out['ordinal_code'] = code
        squared_mass = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 2, 5, 6, 7):
            probability = code.eq(pattern).astype(float).where(triplet_valid).rolling(
                params.complexity_window,
                min_periods=params.complexity_window,
            ).mean()
            out[f'pattern_probability_{pattern}'] = probability
            squared_mass = squared_mass + probability.pow(2)
        out['complexity'] = ((1.0 - squared_mass) * 6.0 / 5.0).clip(0.0, 1.0)
        out['prior_complexity'] = out['complexity'].shift(1)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        valid_bar = (
            np.isfinite(data[['open', 'high', 'low', 'close', 'volume']]).all(axis=1)
            & data['low'].gt(0.0)
            & data['high'].ge(data['low'])
            & close.between(data['low'], data['high'])
            & data['open'].between(data['low'], data['high'])
            & data['volume'].ge(0.0)
        )
        out['valid_bar'] = valid_bar
        out['true_range'] = true_range.where(valid_bar)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window,
        ).mean()
        out['prior_high'] = data['high'].where(valid_bar).rolling(
            params.breakout_window, min_periods=params.breakout_window,
        ).max().shift(1)
        out['breakout_strength'] = (
            (close - out['prior_high']) / out['atr'].where(out['atr'] > 0.0)
        )
        out['complexity_ready'] = out['prior_complexity'].ge(params.complexity_floor)
        out['breakout_ready'] = (
            close.gt(out['prior_high'])
            & out['breakout_strength'].ge(params.breakout_atr)
        )
        out['entry'] = (
            out['complexity_ready'] & out['breakout_ready'] & valid_bar
            & np.isfinite(out['atr']) & out['atr'].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        valid = indicators['valid_bar'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest_close = np.nan
        stop = -np.inf
        for i in range(len(data)):
            usable = valid[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                age += 1
                if not usable:
                    active = False
                else:
                    # The first in-trade close follows the entry decision bar.
                    highest_close = close[i] if np.isnan(highest_close) else max(highest_close, close[i])
                    stop = max(stop, highest_close - params.trail_atr * atr[i])
                    if close[i] <= stop or age >= params.hold_bars:
                        active = False
            elif usable and entry[i]:
                active = True
                age = 0
                highest_close = np.nan
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
        'entry': [True, False, False, False, False, False],
        'valid_bar': True,
    }, index=index)
    data = pd.DataFrame({'close': [100.0, 103.0, 102.0, 101.4, 101.0, 101.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert result['size'].gt(0.0).all()
    data['close'] = 100.0
    indicators['atr'] = 1.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == '__main__':
    _self_check()
