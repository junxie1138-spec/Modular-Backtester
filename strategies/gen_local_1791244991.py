from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BacklogParams:
    pressure_bars: int = 10
    rank_bars: int = 126
    momentum_bars: int = 5
    atr_bars: int = 14
    entry_percentile: float = 0.85
    trail_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.pressure_bars, self.rank_bars, self.momentum_bars, self.atr_bars, self.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between 0.5 and 1')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[BacklogParams]):
    strategy_id = 'gen_local_1791244991'

    @classmethod
    def params_type(cls) -> type[BacklogParams]:
        return BacklogParams

    def warmup_bars(self, params: BacklogParams) -> int:
        return int(max(params.pressure_bars + params.rank_bars + 1,
                       params.momentum_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: BacklogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['candle_range'] = data['high'] - data['low']
        out['close_location'] = ((2.0 * close - data['high'] - data['low'])
                                 / out['candle_range'].replace(0.0, np.nan))
        out.loc[out['candle_range'].eq(0.0), 'close_location'] = 0.0
        out['true_range'] = pd.concat([
            out['candle_range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out['signed_volume'] = out['close_location'] * data['volume']
        out['volume_capacity'] = data['volume'].rolling(params.pressure_bars).sum()
        out['range_capacity'] = out['true_range'].rolling(params.pressure_bars).sum()
        out['pressure'] = (out['signed_volume'].rolling(params.pressure_bars).sum()
                           / out['volume_capacity'].replace(0.0, np.nan))
        out['displacement'] = (close.diff(params.pressure_bars)
                               / out['range_capacity'].replace(0.0, np.nan))
        out['backlog'] = out['pressure'] - out['displacement']
        out['backlog_percentile'] = out['backlog'].rolling(params.rank_bars).rank(pct=True)
        out['momentum'] = close.diff(params.momentum_bars)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['backlog_percentile', 'pressure', 'momentum', 'atr']].notna().all(axis=1)
        ready &= out['atr'].gt(0.0)
        long_entry = (ready & out['backlog_percentile'].ge(params.entry_percentile)
                      & out['pressure'].gt(0.0) & out['momentum'].gt(0.0))
        short_entry = (ready & out['backlog_percentile'].le(1.0 - params.entry_percentile)
                       & out['pressure'].lt(0.0) & out['momentum'].lt(0.0))
        out['candidate'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: BacklogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue

            if position:
                # The entry decision on the previous close fills at this open.
                if age == 0:
                    if not np.isfinite(opens[i]) or i == 0 or not np.isfinite(atrs[i - 1]):
                        position = 0
                        continue
                    extreme = close
                    stop = opens[i] - position * params.trail_atr * atrs[i - 1]
                age += 1
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    # Short positions use the symmetric low-water trailing stop.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
                    continue
            elif confirmed[i] != 0:
                position = int(confirmed[i])
                age = 0
                extreme = np.nan
                stop = np.nan

            signals[i] = position

        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7, freq='B')
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 1, 3, 2, 0, -1], dtype=float)
        sample = pd.DataFrame({'open': close, 'high': close + 1,
                               'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'candidate': [0, direction, direction, 0, 0, 0, 0],
                                'atr': [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0]}, index=index)
        result = strategy.generate_signals(sample, derived, None, BacklogParams(trail_atr=2.0)).data
        assert result['signal'].tolist() == [0, 0, 0, direction, direction, direction, 0]
        expired = strategy.generate_signals(sample, derived, None, BacklogParams(trail_atr=20.0, hold_bars=2)).data
        assert expired['signal'].tolist() == [0, 0, 0, direction, direction, 0, 0]
        assert result['size'].gt(0.0).all()
