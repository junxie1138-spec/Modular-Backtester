from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeChangeParams:
    baseline_bars: int = 30
    recent_bars: int = 3
    change_threshold: float = 2.0
    close_threshold: float = 0.65
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeChangeParams]):
    strategy_id = 'gen_local-w2_1791265751'

    @classmethod
    def params_type(cls) -> type[RangeChangeParams]:
        return RangeChangeParams

    @staticmethod
    def warmup_bars(params: RangeChangeParams) -> int:
        if params.baseline_bars < 2 or params.recent_bars < 2 or params.hold_bars < 1:
            raise ValueError('Lookbacks must be at least two and holding duration positive.')
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError('Change threshold must be finite and positive.')
        if not 0.5 < params.close_threshold < 1.0:
            raise ValueError('Close threshold must lie between 0.5 and 1.')
        return int(max(200, params.baseline_bars + params.recent_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RangeChangeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['width'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['relative_range'] = out['width'].div(data['close'].where(data['close'] > 0))
        out['close_location'] = (data['close'] - data['low']).div(out['width'].where(out['width'] > 0)).clip(0.0, 1.0)
        out['recent_range'] = out['relative_range'].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out['baseline_range'] = out['relative_range'].rolling(params.baseline_bars, min_periods=params.baseline_bars).mean().shift(params.recent_bars)
        out['baseline_std'] = out['relative_range'].rolling(params.baseline_bars, min_periods=params.baseline_bars).std(ddof=0).shift(params.recent_bars)
        out['change_score'] = (out['recent_range'] - out['baseline_range']).div(out['baseline_std'].where(out['baseline_std'] > 0))
        out['previous_score'] = out['change_score'].shift(1)
        out['recent_location'] = out['close_location'].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out[['change_score', 'previous_score', 'recent_location', 'ma200']].notna().all(axis=1)
        out['changepoint'] = out['ready'] & out['change_score'].ge(params.change_threshold) & out['previous_score'].lt(params.change_threshold)
        out['long_entry'] = out['changepoint'] & out['recent_location'].ge(params.close_threshold) & data['close'].gt(out['ma200'])
        out['short_entry'] = out['changepoint'] & out['recent_location'].le(1.0 - params.close_threshold) & data['close'].lt(out['ma200'])
        out['entry_direction'] = out['long_entry'].astype(int) - out['short_entry'].astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeChangeParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RangeChangeParams()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry_direction': np.zeros(24, dtype=int)}, index=index)
    indicators.iloc[[0, 3, 8, 10, 12], 0] = [1, -1, -1, -1, 1]
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 2 + [-1] * 8 + [0] * 5
    assert result['size'].gt(0).all()
    assert strategy.indicators(data, params)['entry_direction'].eq(0).all()
    assert strategy.warmup_bars(params) >= 200
