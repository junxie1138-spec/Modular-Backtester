from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_bars: int = 20
    streak_bars: int = 3
    edge_threshold: float = 0.75
    wick_bars: int = 3
    atr_bars: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1790976156'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.channel_bars + params.streak_bars - 1,
                       params.wick_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.channel_bars, params.streak_bars, params.wick_bars,
               params.atr_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.5 < params.edge_threshold < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Invalid edge threshold or stop multiplier.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['channel_high'] = data['high'].rolling(params.channel_bars).max()
        out['channel_low'] = data['low'].rolling(params.channel_bars).min()
        width = (out['channel_high'] - out['channel_low']).where(lambda x: x > 0)
        out['relative_position'] = (close - out['channel_low']) / width
        upper = out['relative_position'].ge(params.edge_threshold)
        lower = out['relative_position'].le(1.0 - params.edge_threshold)
        out['upper_streak'] = upper.astype(int).groupby((~upper).cumsum()).cumsum()
        out['lower_streak'] = lower.astype(int).groupby((~lower).cumsum()).cumsum()
        out['lower_wick'] = (data[['open', 'close']].min(axis=1) - data['low']).clip(lower=0)
        out['upper_wick'] = (data['high'] - data[['open', 'close']].max(axis=1)).clip(lower=0)
        out['bar_range'] = (data['high'] - data['low']).where(lambda x: x > 0)
        out['wick_imbalance'] = (out['lower_wick'] - out['upper_wick']) / out['bar_range']
        out['weighted_wick'] = out['wick_imbalance'] * data['volume'].clip(lower=0)
        out['window_volume'] = data['volume'].clip(lower=0).rolling(params.wick_bars).sum()
        out['wick_pressure'] = out['weighted_wick'].rolling(params.wick_bars).sum() / out['window_volume'].where(lambda x: x > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out['ready'] = np.isfinite(out[['relative_position', 'wick_pressure', 'atr']]).all(axis=1) & out['atr'].gt(0)
        out['long_entry'] = out['ready'] & out['upper_streak'].ge(params.streak_bars) & out['wick_pressure'].gt(0)
        out['short_entry'] = out['ready'] & out['lower_streak'].ge(params.streak_bars) & out['wick_pressure'].lt(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = (closes[i] < fixed_stop if position == 1 else closes[i] > fixed_stop)
                expired = i - entry_bar + 1 >= params.hold_bars
                if not np.isfinite(closes[i]) or stopped or expired:
                    position = 0
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i] or shorts[i]:
                    position = 1 if longs[i] else -1
                    entry_bar = i + 1
                    entry_atr = atr[i]
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0,
                         'close': [100., 100., 97., 100., 100., 100., 103., 100.],
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0,
                               'long_entry': [True, False, False, False, False, False, False, False],
                               'short_entry': [False, False, False, False, True, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, Params())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    timed = strategy.generate_signals(data.assign(close=100.0), indicators, ctx, Params(hold_bars=2))
    assert timed.data['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
