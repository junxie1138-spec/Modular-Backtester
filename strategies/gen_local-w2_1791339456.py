from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EscapeParams:
    ma_window: int = 20
    percentile_window: int = 126
    escape_percentile: float = 0.85
    crossing_window: int = 10
    min_crossings: int = 3
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[EscapeParams]):
    strategy_id = 'gen_local-w2_1791339456'

    @classmethod
    def params_type(cls) -> type[EscapeParams]:
        return EscapeParams

    @staticmethod
    def warmup_bars(params: EscapeParams) -> int:
        if params.ma_window < 2 or params.percentile_window < 2:
            raise ValueError('MA and percentile windows must be at least two.')
        if params.crossing_window < 1 or not 1 <= params.min_crossings <= params.crossing_window:
            raise ValueError('Invalid crossing requirements.')
        if not 0.0 < params.escape_percentile < 1.0:
            raise ValueError('Percentile must lie strictly between zero and one.')
        if params.hold_bars not in (3, 4, 5):
            raise ValueError('Holding period must be three to five bars.')
        return int(params.ma_window + max(params.percentile_window, params.crossing_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: EscapeParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['mean'] = out['log_close'].rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['std'] = out['log_close'].rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = ((out['log_close'] - out['mean']) / out['std'].where(out['std'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['abs_z'] = out['z'].abs()
        out['prior_z'] = out['z'].shift(1)
        out['prior_abs_z'] = out['abs_z'].shift(1)
        out['threshold'] = out['prior_abs_z'].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.escape_percentile)
        out['prior_threshold'] = out['threshold'].shift(1)
        valid_pair = out['z'].notna() & out['prior_z'].notna()
        out['crossing'] = (np.sign(out['z']) * np.sign(out['prior_z']) < 0).astype(float).where(valid_pair)
        out['prior_crossing'] = out['crossing'].shift(1)
        out['crossing_count'] = out['prior_crossing'].rolling(params.crossing_window, min_periods=params.crossing_window).sum()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EscapeParams) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            (indicators['abs_z'] > indicators['threshold'])
            & (indicators['prior_abs_z'] <= indicators['prior_threshold'])
            & (indicators['abs_z'] > indicators['prior_abs_z'])
            & (indicators['crossing_count'] >= params.min_crossings)
        ).fillna(False)
        entries = np.where(eligible, np.sign(indicators['z'].fillna(0.0)), 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = EscapeParams(hold_bars=3)
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    z = pd.Series([np.nan, 0.4, 2.0, 3.0, -3.0, 0.0, -2.0, -3.0, 4.0, 3.0, 2.0, 0.0], index=index)
    synthetic = pd.DataFrame({'z': z, 'abs_z': z.abs(), 'prior_abs_z': z.abs().shift(1), 'threshold': 1.0, 'prior_threshold': 1.0, 'crossing_count': 3.0}, index=index)
    result = strategy.generate_signals(data, synthetic, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1, -1, 0, 0]
    assert result['size'].gt(0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data['signal'].eq(0).all()
