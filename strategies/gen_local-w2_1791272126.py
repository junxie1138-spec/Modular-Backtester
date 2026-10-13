from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    expansion_ratio: float = 1.20
    contraction_ratio: float = 0.85
    shock_atr: float = 0.80
    trail_atr: float = 1.25
    hold_bars: int = 2

    def __post_init__(self):
        values = [self.expansion_ratio, self.contraction_ratio, self.shock_atr, self.trail_atr]
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Parameters must be finite')
        if not 0 < self.contraction_ratio < self.expansion_ratio:
            raise ValueError('Volatility thresholds must be positive and ordered')
        if self.shock_atr <= 0 or self.trail_atr <= 0:
            raise ValueError('Shock and trailing multiples must be positive')
        if type(self.hold_bars) is not int or self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791272126'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['fast_atr'] = out['true_range'].rolling(5, min_periods=5).mean()
        out['slow_atr'] = out['true_range'].rolling(40, min_periods=40).mean().shift(1)
        out['vol_ratio'] = out['fast_atr'] / out['slow_atr'].where(out['slow_atr'] > 0)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['shock'] = data['close'].diff() / out['atr'].shift(1).where(out['atr'].shift(1) > 0)
        regime = pd.Series(np.nan, index=data.index)
        regime.loc[out['vol_ratio'] >= params.expansion_ratio] = 1.0
        regime.loc[out['vol_ratio'] <= params.contraction_ratio] = -1.0
        out['regime'] = regime.ffill().fillna(0.0)
        candidate = np.sign(out['shock']) * out['regime']
        valid = np.isfinite(out[['atr', 'vol_ratio', 'shock', 'ma200']]).all(axis=1)
        valid &= (out['atr'] > 0) & (out['shock'].abs() >= params.shock_atr)
        long_entry = valid & (candidate > 0) & (data['close'] > out['ma200'])
        short_entry = valid & (candidate < 0) & (data['close'] < out['ma200'])
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        highest_close = -np.inf
        lowest_close = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)

        for i, (close, atr, entry) in enumerate(zip(closes, atrs, entries)):
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                signals[i] = 0
                continue
            if position:
                # Entry fills on this bar after the final signal shift.
                # Only closes observed since that fill update the watermark.
                highest_close = max(highest_close, close)
                lowest_close = min(lowest_close, close)
                trailing_floor = max(trailing_floor, highest_close - params.trail_atr * atr)
                short_ceiling = min(short_ceiling, lowest_close + params.trail_atr * atr)
                # The required highest-close trail applies to both directions;
                # for shorts it takes profits. The ceiling limits adverse moves.
                stopped = close < trailing_floor
                stopped |= position == -1 and close > short_ceiling
                if stopped or i - entry_bar >= params.hold_bars:
                    position = 0
                signals[i] = position
                # An exit decision cannot reopen a position on the same close.
                continue
            if entry in (-1, 1):
                position = int(entry)
                entry_bar = i
                highest_close = -np.inf
                lowest_close = np.inf
                trailing_floor = -np.inf
                short_ceiling = np.inf
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        derived = pd.DataFrame({'atr': 1.0, 'entry': [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, Params()).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]
        assert (result['size'] > 0).all()
    derived = strategy.indicators(data, Params())
    assert derived['entry'].eq(0).all()
    assert strategy.warmup_bars(Params()) >= 200


if __name__ == '__main__':
    _self_check()
