from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 4
    percentile_bars: int = 80
    acceleration_quantile: float = 0.85
    hold_bars: int = 2

    def __post_init__(self):
        if self.roc_bars < 2 or self.percentile_bars < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < self.acceleration_quantile < 1.0:
            raise ValueError('Acceleration quantile must be between zero and one.')
        if self.hold_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791324009'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.roc_bars + params.percentile_bars + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        opening = data['open'].where(data['open'] > 0)
        closing = data['close'].where(data['close'] > 0)
        out['session_log_return'] = np.log(closing / opening)
        out['session_log_return'] = out['session_log_return'].replace([np.inf, -np.inf], np.nan)
        out['session_roc'] = np.expm1(out['session_log_return'].rolling(params.roc_bars, min_periods=params.roc_bars).sum())
        out['session_roc'] = out['session_roc'].replace([np.inf, -np.inf], np.nan)
        out['acceleration'] = out['session_roc'].diff()
        out['acceleration'] = out['acceleration'].replace([np.inf, -np.inf], np.nan)
        out['close_roc'] = closing.pct_change(periods=params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['acceleration_threshold'] = out['acceleration'].abs().shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.acceleration_quantile)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators[['session_roc', 'acceleration', 'close_roc', 'acceleration_threshold']].notna().all(axis=1)
        extreme = indicators['acceleration'].abs() > indicators['acceleration_threshold']
        long_entry = ready & extreme & (indicators['acceleration'] > 0) & (indicators['session_roc'] > 0) & (indicators['close_roc'] > 0)
        short_entry = ready & extreme & (indicators['acceleration'] < 0) & (indicators['session_roc'] < 0) & (indicators['close_roc'] < 0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signals[i] = position
            elif candidate:
                position = int(candidate)
                exit_bar = i + params.hold_bars
                signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    direction = pd.Series([1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0], index=index)
    indicators = pd.DataFrame({'session_roc': direction, 'acceleration': direction, 'close_roc': direction, 'acceleration_threshold': 0.5}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result['size'].gt(0).all()
