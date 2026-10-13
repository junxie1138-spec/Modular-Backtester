from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCoreParams:
    lookback: int = 5
    max_span: float = 1.6
    min_gap: float = 0.25
    hold_bars: int = 4

    def __post_init__(self):
        if not isinstance(self.lookback, int) or not 2 <= self.lookback <= 9:
            raise ValueError('lookback must be an integer from 2 to 9')
        if not isinstance(self.hold_bars, int) or not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be an integer from 3 to 5')
        if not np.isfinite(self.max_span) or self.max_span < 1.0:
            raise ValueError('max_span must be finite and at least 1')
        if not np.isfinite(self.min_gap) or self.min_gap <= 0:
            raise ValueError('min_gap must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapCoreParams]):
    strategy_id = 'gen_local-w2_1791316896'

    @classmethod
    def params_type(cls) -> type[GapCoreParams]:
        return GapCoreParams

    def warmup_bars(self, params: GapCoreParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: GapCoreParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        n = params.lookback
        high = data['high'].replace([np.inf, -np.inf], np.nan)
        low = data['low'].replace([np.inf, -np.inf], np.nan)
        daily_range = (high - low).where(high > low)
        out['mean_range'] = daily_range.rolling(n, min_periods=n).mean().shift(1)
        out['envelope_high'] = high.rolling(n, min_periods=n).max().shift(1)
        out['envelope_low'] = low.rolling(n, min_periods=n).min().shift(1)
        out['core_high'] = high.rolling(n, min_periods=n).min().shift(1)
        out['core_low'] = low.rolling(n, min_periods=n).max().shift(1)
        out['core_mid'] = (out['core_high'] + out['core_low']) / 2.0
        out['span_ratio'] = (out['envelope_high'] - out['envelope_low']) / out['mean_range']
        out['gap_units'] = (data['open'] - data['close'].shift(1)) / out['mean_range']
        out = out.replace([np.inf, -np.inf], np.nan)
        valid_bar = np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
        out['entry'] = (
            valid_bar
            & out['mean_range'].gt(0)
            & out['core_high'].gt(out['core_low'])
            & out['span_ratio'].le(params.max_span)
            & out['gap_units'].le(-params.min_gap)
            & data['open'].lt(out['envelope_low'])
            & data['close'].ge(out['core_mid'])
            & data['close'].le(out['core_high'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapCoreParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if i < exit_bar:
                signals[i] = 1
            elif i == exit_bar:
                # Force a flat target on the scheduled exit, even if entry is true.
                continue
            elif entry:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=18)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    for hold in (3, 4, 5):
        params = GapCoreParams(hold_bars=hold)
        entries = pd.DataFrame({'entry': True}, index=index)
        result = strategy.generate_signals(data, entries, None, params).data
        expected = [0] + [int(i % (hold + 1) < hold) for i in range(len(data) - 1)]
        assert result['signal'].tolist() == expected
        assert result['size'].gt(0).all()
        assert strategy.warmup_bars(params) <= 10
    params = GapCoreParams()
    indicators = strategy.indicators(data, params)
    assert not indicators['entry'].any()
    event = data.copy()
    event.iloc[5, event.columns.get_loc('open')] = 98.5
    assert strategy.indicators(event, params)['entry'].iloc[5]
    event.iloc[5, event.columns.get_loc('close')] = np.nan
    assert not strategy.indicators(event, params)['entry'].iloc[5]


if __name__ == '__main__':
    _self_check()
