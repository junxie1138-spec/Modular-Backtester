from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapIslandParams:
    atr_period: int = 14
    channel_period: int = 15
    percentile_window: int = 100
    gap_percentile: float = 0.90
    stop_atr: float = 2.0
    max_hold: int = 8

    def __post_init__(self):
        if min(self.atr_period, self.channel_period, self.percentile_window, self.max_hold) < 1:
            raise ValueError('Lookbacks and holding period must be positive')
        if not 0.0 < self.gap_percentile < 1.0:
            raise ValueError('gap_percentile must be between zero and one')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapIslandParams]):
    strategy_id = 'gen_local_1791328603'

    @classmethod
    def params_type(cls) -> type[GapIslandParams]:
        return GapIslandParams

    def warmup_bars(self, params: GapIslandParams) -> int:
        return int(max(params.atr_period + params.percentile_window + 1,
                       params.channel_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapIslandParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        prior_atr = out['atr'].shift(1).where(out['atr'].shift(1) > 0.0)
        out['normalized_gap'] = ((data['open'] - previous_close) / prior_atr).replace([np.inf, -np.inf], np.nan)
        out['gap_cutoff'] = out['normalized_gap'].abs().rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.gap_percentile).shift(1)
        out['previous_high'] = data['high'].rolling(
            params.channel_period, min_periods=params.channel_period
        ).max().shift(1)
        out['previous_low'] = data['low'].rolling(
            params.channel_period, min_periods=params.channel_period
        ).min().shift(1)
        ready = out.notna().all(axis=1) & out['atr'].gt(0.0)
        extreme = out['normalized_gap'].abs().gt(out['gap_cutoff'])
        upward = ready & extreme & out['normalized_gap'].gt(0.0) & data['low'].gt(out['previous_high'])
        downward = ready & extreme & out['normalized_gap'].lt(0.0) & data['high'].lt(out['previous_low'])
        out['entry'] = np.select([upward, downward], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapIslandParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        stop_price = np.nan
        held = 0

        for i in range(len(data)):
            if position != 0:
                # A prior-close decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    stop_price = entry_price - position * params.stop_atr * atr[i - 1]
                    held = 0
                held += 1
                invalid = not (np.isfinite(entry_price) and np.isfinite(stop_price) and np.isfinite(closes[i]))
                stopped = (position == 1 and closes[i] < stop_price) or (position == -1 and closes[i] > stop_price)
                if invalid or stopped or held >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    stop_price = np.nan
                    held = 0
            elif entries[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(entries[i])
                entry_price = np.nan
            targets[i] = position

        df = pd.DataFrame(index=data.index)
        df['signal'] = targets
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5, freq='D')
    for direction in (1, -1):
        # ATR changes after entry must never move the fixed stop.
        data = pd.DataFrame({
            'open': [100.0, 110.0, 110.0, 110.0, 110.0],
            'close': [100.0, 110.0 - direction * 4.0,
                      110.0 - direction * 5.0, 110.0, 110.0],
        }, index=index)
        derived = pd.DataFrame({'entry': [direction, 0, 0, 0, 0],
                                'atr': [2.0, 20.0, 20.0, 20.0, 20.0]}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, derived, None, GapIslandParams())
        assert result.data['signal'].tolist() == [0, direction, direction, 0, 0]
        data['close'] = 110.0
        result = GeneratedStrategy.generate_signals(None, data, derived, None, GapIslandParams(max_hold=2))
        assert result.data['signal'].tolist() == [0, direction, direction, 0, 0]
        derived['entry'] = 0
        result = GeneratedStrategy.generate_signals(None, data, derived, None, GapIslandParams())
        assert result.data['signal'].eq(0).all()
        assert result.data['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
