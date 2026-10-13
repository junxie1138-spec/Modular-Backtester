from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MonthOpeningParams:
    volume_window: int = 20
    percentile_window: int = 126
    volume_quantile: float = 0.75
    seasonal_sessions: int = 3
    atr_window: int = 14
    atr_multiple: float = 1.0
    hold_bars: int = 2

    def __post_init__(self):
        if min(self.volume_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.volume_quantile < 1.0:
            raise ValueError('volume_quantile must lie strictly between zero and one.')
        if not 1 <= self.seasonal_sessions <= 5:
            raise ValueError('seasonal_sessions must be between one and five.')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive.')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')


class GeneratedStrategy(BaseStrategy[MonthOpeningParams]):
    strategy_id = 'gen_local_1791290298'

    @classmethod
    def params_type(cls):
        return MonthOpeningParams

    @staticmethod
    def warmup_bars(params: MonthOpeningParams) -> int:
        return int(max(params.volume_window + params.percentile_window + 1,
                       params.atr_window + 1, 3))

    @staticmethod
    def indicators(data: pd.DataFrame, params: MonthOpeningParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['move'] = data['close'].diff()
        out['previous_move'] = out['move'].shift(1)
        out['volume_baseline'] = data['volume'].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out['relative_volume'] = data['volume'].div(
            out['volume_baseline'].where(out['volume_baseline'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['volume_threshold'] = out['relative_volume'].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.volume_quantile).shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        month_key = data.index.year * 12 + data.index.month
        out['month_session'] = pd.Series(1, index=data.index).groupby(
            month_key
        ).cumsum()
        out['seasonal_window'] = (
            (out['month_session'] <= params.seasonal_sessions)
            & (data.index.day <= 7)
        )
        ready = np.isfinite(out[[
            'move', 'previous_move', 'relative_volume', 'volume_threshold', 'atr'
        ]]).all(axis=1)
        out['entry'] = (
            ready
            & out['seasonal_window']
            & (out['previous_move'] < 0)
            & (out['move'] > 0)
            & (data['close'] > data['open'])
            & (out['relative_volume'] > out['volume_threshold'])
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: MonthOpeningParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                continue
            if active:
                # The first held close occurs one bar after the entry decision.
                age += 1
                peak = max(peak, close)
                stop = max(stop, peak - params.atr_multiple * atr)
                if close <= stop or age >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i]:
                active = True
                age = 0
                peak = stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5, freq='D')
    strategy = GeneratedStrategy()
    params = MonthOpeningParams()
    indicators = pd.DataFrame({
        'atr': [1.0] * 5,
        'entry': [True, False, False, False, False],
    }, index=index)
    for prices in ([100, 101, 99, 100, 100], [100, 101, 102, 103, 104]):
        data = pd.DataFrame({'close': prices}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 1, 1, 0, 0]
        assert (result['size'] > 0).all()
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    params.hold_bars = 2
    indicators['entry'] = False
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
